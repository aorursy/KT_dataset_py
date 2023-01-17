import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/mm_master_demos.csv', index_col=0)
print(df.columns)
print(df.shape)
CONST_VALID_MAPS = ['de_dust2', 'de_cache', 'de_inferno', 'de_mirage', 'de_cbble', 'de_overpass', 'de_train', 'de_nuke']
CONST_VALID_RANKS = list(range(6,17)) # SEM to LEM
df = df[df.map.isin(CONST_VALID_MAPS) & df.avg_match_rank.isin(CONST_VALID_RANKS)]
df_recv = df.groupby(['file', 'map', 'round', 'vic_id', 'vic_side']).agg({'hp_dmg': 'sum',  'seconds': 'last'}).reset_index()
kill_key = df_recv.loc[df_recv.hp_dmg >= 100, ['file', 'map', 'round', 'vic_id', 'vic_side', 'seconds']]
df_kill = pd.merge(df, kill_key, on = list(kill_key.columns))
df_kill = df_kill.groupby(['file', 'map', 'round', 'vic_id', 'vic_side', 'seconds']).last().reset_index()
df_kill = df_kill[df_kill.att_side.isin(['CounterTerrorist', 'Terrorist'])] # remove non-player kills
df_kill['att_flg'] = df_kill['att_side'].map({'CounterTerrorist': 1, 'Terrorist': -1}) # helps maps sides for features matrix

df_kill['target'] = df_kill['winner_side'] == df_kill['att_side'] # target var
df_kill.loc[:, 'equip_val_adv'] = (df_kill['ct_eq_val']/df_kill['t_eq_val']-1)*df_kill['att_flg']
# alternative, categorical round type
df_kill['cat_round_type'] = df_kill.apply(lambda x: f"ADV {x.round_type}" if x['equip_val_adv'] > 0 else x.round_type, axis = 1)
df_kill['died_to_ct'] = df_kill['vic_side'] == 'Terrorist'
df_kill['died_to_t'] = ~df_kill['died_to_ct']
df_kill[['died_to_ct', 'died_to_t']] = df_kill.groupby(['file', 'map', 'round'])[['died_to_ct', 'died_to_t']].apply(lambda x: x.cumsum())
df_kill['man_adv'] = ((df_kill['died_to_ct'] - df_kill['died_to_t'])*df_kill['att_flg']).astype(int)
df_kill['ct_alive'] = 5 - df_kill['died_to_t']
df_kill['t_alive'] = 5 - df_kill['died_to_ct']
df_kill['matchup'] = df_kill.apply(lambda x: f"{x['ct_alive']:.0f}v{x['t_alive']:.0f}" if x['att_flg'] == True else f"{x['t_alive']}v{x['ct_alive']}", axis = 1)
round_start_mat = []
for k,g in df_kill.groupby(['file', 'map', 'round', 'cat_round_type', 'winner_side']):
    val_adv = g.groupby('att_side')['equip_val_adv'].mean()
    # incase of no kills on either side
    if 'Terrorist' not in val_adv.index:
        val_adv['Terrorist'] = -val_adv['CounterTerrorist']
    elif 'CounterTerrorist' not in val_adv.index:
        val_adv['CounterTerrorist'] = -val_adv['Terrorist']
    round_start_mat.append(pd.Series([k[1],'CounterTerrorist', False, k[3], k[4] == 'CounterTerrorist', val_adv['CounterTerrorist'], 0, '5v5'], index=['map', 'att_side', 'is_bomb_planted', 'cat_round_type', 'target', 'equip_val_adv', 'man_adv', 'matchup']))
    round_start_mat.append(pd.Series([k[1],'Terrorist', False, k[3], k[4] == 'Terrorist', val_adv['Terrorist'], 0, '5v5'], index=['map', 'att_side', 'is_bomb_planted', 'cat_round_type', 'target', 'equip_val_adv', 'man_adv', 'matchup']))

round_start_mat = pd.DataFrame(round_start_mat)
df_kill = df_kill.append(round_start_mat)
features = ['map', 'att_side', 'is_bomb_planted', 'cat_round_type', 'equip_val_adv', 'man_adv', 'matchup']
y_target = 'target'

df_x = df_kill[features+[y_target]]
df_x = df_x[df_x['matchup'].isin(['5v5','5v4', '4v4', '4v3', '3v3', '2v3', '3v1', '1v2', '2v0', '5v3',
       '5v2', '5v1', '1v4', '1v3', '3v0', '4v2', '4v1', '0v1', '2v2',
       '2v1', '4v0', '1v1', '1v0', '5v0', '0v2', '3v2', '3v4', '0v3',
       '2v4', '0v4', '4v5', '2v5', '0v5', '3v5', '0v0', '1v5'])] # remove extra player anomaly matches
print(df_x.shape)
df_x.head()
from sklearn.model_selection import train_test_split

x = pd.get_dummies(df_x[features])
y = df_x[y_target]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, train_size = 0.8, random_state = 1337)
print(x_train.shape)
print(x_test.shape)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
def model_it(model):
    model.fit(x_train, y_train)
    print('Training Accuracy: %.7f    |    Test Accuracy: %.7f' % (model.score(x_train, y_train), model.score(x_test, y_test)))
    plot_roc_curve(y_test, model.predict_proba(x_test)[:,1])
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model_it(model)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, max_depth = 15, min_samples_split = 10)

model_it(model)
df_sample_killers = pd.concat([ pd.DataFrame(['CounterTerrorist'], columns=['att_side']), df_kill[['att_id', 'vic_id', 'wp_type', 'map', 'att_side', 'vic_side', 'is_bomb_planted', 'matchup']].iloc[:8]])
df_sample_x = x[:8]
init_round = x.iloc[[0]]
init_round['matchup_5v4'] = 0
init_round['matchup_5v5'] = 1
df_sample_x = pd.concat([init_round, df_sample_x])
df_sample_killers['win_prob_attacker'] = model.predict_proba(df_sample_x)[:,1]
df_sample_killers['win_prob_CT'] = df_sample_killers.apply(lambda x: x['win_prob_attacker'] if x['att_side'] == 'CounterTerrorist' else 1. - x['win_prob_attacker'], axis=1)
df_sample_killers['win_prob_T'] = 1. - df_sample_killers['win_prob_CT']
df_sample_killers['delta_CT'] = df_sample_killers['win_prob_CT'] - df_sample_killers['win_prob_CT'].shift(1)
df_sample_killers['delta_T'] = df_sample_killers['win_prob_T'] - df_sample_killers['win_prob_T'].shift(1)
df_sample_killers['PRI'] = df_sample_killers.apply(lambda x: x['delta_CT'] if x['att_side'] == 'CounterTerrorist' else x['delta_T'], axis=1)
df_sample_killers.att_id = df_sample_killers.att_id.astype(str)
df_sample_killers
df_sample_killers.dropna().groupby(['att_id', 'att_side'])['PRI'].sum()
