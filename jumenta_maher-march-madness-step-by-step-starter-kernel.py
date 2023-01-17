import matplotlib.pyplot as plt

import numpy as np # linear algebra

import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss

import warnings

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from tpot import TPOTClassifier

import glob

print(os.listdir('../input/daily-rpi-2019'))
ipfl = os.listdir("../input/mens-machine-learning-competition-2019")

ipfls = [x for x in ipfl if '.' in x]

ipfds = [x for x in ipfl if '.' not in x]

print(f'Files in ../input: {ipfls}\n')

print(f'Folders in ../input: {ipfds}\n')

pbpfl = os.listdir("../input/mens-machine-learning-competition-2019/playbyplay_2012")

print(f'Files in representative playbyplay folder: {pbpfl}\n')

dfs = os.listdir("../input/mens-machine-learning-competition-2019/datafiles")

print(f'Files in datafiles folder: {dfs}\n')

mofl = os.listdir("../input/mens-machine-learning-competition-2019/masseyordinals")

print(f'Files in masseyordinals folder: {mofl}')



st1_tr =  pd.read_csv("../input/daily-rpi-2019/train_stage1_Prepped.csv")

st1_tr.head()

st1_tt =  pd.read_csv("../input/daily-rpi-2019/test_stage1_Prepped.csv")

st1_tt.head()
sub = pd.read_csv('../input/mens-machine-learning-competition-2019/SampleSubmissionStage1.csv')

sub.tail(3)

pipeline_optimizer = TPOTClassifier()

pipeline_optimizer = TPOTClassifier(generations=10, population_size=200, cv=5,

                                    random_state=42, verbosity=2)
fl_2014 = '../input/mens-machine-learning-competition-2019/playbyplay_2014/'

events = pd.read_csv(f'{fl_2014}Events_2014.csv')

players = pd.read_csv(f'{fl_2014}Players_2014.csv')

print(f'The events dataframe contains data from the Events_2014.csv file and is {events.shape[0]}'

      f' rows by {events.shape[1]} \ncolumns. The players dataframe contains the data from '

      f'the Players_2014.csv file and is {players.shape[0]} rows by {players.shape[1]} \nrows.')
events.head(3)
players.head(3)
## The pandas Series.value_counts() method returns a pd.Sereis object with value counts for each unique

## value in a pd.Series. 



# seperate data into winners and loosers

winners = events[events['EventTeamID'] == events['WTeamID']] 

loosers = events[events['EventTeamID'] != events['WTeamID']]

val_wnrs = winners.EventType.value_counts().sort_values(ascending=True).to_frame()

val_lsrs = loosers.EventType.value_counts().sort_values(ascending=True).to_frame()



# stack the data for plotting with seaborn

totals = val_wnrs.merge(val_lsrs, right_index=True, left_index=True, suffixes=['_w', '_l']

                       ).sort_values(by='EventType_w', ascending=False)

totals.rename(columns={'EventType_w':'winner', 'EventType_l':'looser'}, inplace=True)

tts = totals.stack().reset_index()

tts.columns = ['event', 'win', 'count']



## plot data

plt.figure(figsize=(10,7))

sns.barplot(x='count', y='event', hue='win', data=tts, orient='h')

plt.title('2014 Events by Winners and Loosers', size=15)

plt.show()
df_fdr = '../input/mens-machine-learning-competition-2019/datafiles/'
files = os.listdir(df_fdr)

names = [x.split('.')[0] for x in files]

path_list = [f'{df_fdr}{x}' for x in files]

df_dict = {}

for num, name in enumerate(names):

    try:

        df_dict[name] = pd.read_csv(path_list[num])

    except:

        print(f'{name} did not load')
print('Summary of files:\n___________________________')

for name, df in df_dict.items():

    print(f'{name}: {df.shape}')
seeds = df_dict.pop('NCAATourneySeeds') # This can only be run once.

seeds['seed_num'] = seeds.Seed.apply(lambda x: int(x[1:3]))

seeds['seed_letter'] = seeds.Seed.apply(lambda x: x[0])

seeds.head(3)
tcr = df_dict.pop('NCAATourneyCompactResults') # This can only be run once.

tcr['Team1'] = np.where((tcr.WTeamID < tcr.LTeamID), tcr.WTeamID, tcr.LTeamID)

tcr['Team2'] = np.where((tcr.WTeamID > tcr.LTeamID), tcr.WTeamID, tcr.LTeamID)

tcr['target'] = np.where((tcr['WTeamID'] < tcr['LTeamID']),1,0)

tcr.head(3)
tcr_grouped = tcr.groupby('WTeamID')['Season'].count()

winners = tcr_grouped.sort_values(ascending=False).iloc[:20]
teams = df_dict.pop("Teams")

winners = winners.to_frame().merge(teams[['TeamID', 'TeamName']], how='left', left_index=True, right_on='TeamID')
plt.figure(figsize=(10,6))

sns.barplot(y=winners.TeamName, x=winners.Season, orient='h', color='b')

plt.xlabel('Wins')

plt.ylabel('Team Name')

plt.title('Most Wins since 1985', size=20)

plt.show()
def add_features(df, seeds=seeds):

    df = df.merge(seeds, how='left', left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'])

    df = df.merge(seeds, how='left', left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], 

          suffixes=('Team1', 'Team2'))

    

    # encode the seed letters for both teams using the same LabelEncoder

    lb = LabelEncoder()

    df['seed_letterTeam1'] = lb.fit_transform(df['seed_letterTeam1'])

    df['seed_letterTeam2'] = lb.transform(df['seed_letterTeam2'])

    return df
tcr_ = add_features(tcr)
tcr_.head(3)
features = ['seed_numTeam1', 'seed_letterTeam1', 'seed_numTeam2', 'seed_letterTeam2']

X = tcr_[features]

y = tcr_['target']
# split model for train and test

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=7)
xgb = XGBClassifier() 

xgb.fit(X_train, y_train) # train

preds = xgb.predict_proba(X_test) # use predict_proba or a wrong prediction will result in infinity









#important_features = pd.Series(data=tcr_.feature_importances_,index=X.columns)

#important_features.sort_values(ascending=False,inplace=True)
st1_tr =  pd.read_csv("../input/daily-rpi-2019/train_stage1_Prepped.csv")

st1_tr.head()



features2 = ['PPG_1', 'OPPG_1', 'TOPG_1', 'FG3_PCT_1', 'AEE_1', 'PPG_2', 'OPPG_2', 'TOPG_2', 'FG3_PCT_2', 'AEE_2', 'seed_diff']

X2 = st1_tr[features2]

y2 = st1_tr['target']



#pipeline_optimizer.fit(X2, y2)

#print(pipeline_optimizer.score(X2, y2))





from sklearn.svm import LinearSVC

from sklearn.datasets import make_classification

from sklearn.preprocessing import Normalizer

from sklearn.calibration import CalibratedClassifierCV





transformer = Normalizer().fit(X2) # fit does nothing.

X2 = transformer.transform(X2)



#clf = LinearSVC(Normalizer(input_matrix, norm=l2), C=10.0, dual=False, loss=squared_hinge, penalty=l2, tol=0.001)

#clf = LinearSVC(C=10.0, dual=False, loss='squared_hinge', penalty='l2', tol=0.001)

clf = LinearSVC(C=10.0, dual=False, loss='squared_hinge', penalty='l2', tol=0.001)

clf2 = CalibratedClassifierCV(clf) 

clf2.fit(X2, y2)

#clf.fit(Normalizer(X2, y2, norm='l2'), C=10.0, dual=False, loss='squared_hinge', penalty='l2', tol=0.001)

#print(clf2.coef_)



st1_tt =  pd.read_csv("../input/daily-rpi-2019/test_stage1_Prepped.csv")

st1_tt.head()

features2 = ['PPG_1', 'OPPG_1', 'TOPG_1', 'FG3_PCT_1', 'AEE_1', 'PPG_2', 'OPPG_2', 'TOPG_2', 'FG3_PCT_2', 'AEE_2', 'seed_diff']

X2_test = st1_tt[features2]

y2_test = st1_tt['target']



transformer = Normalizer().fit(X2_test) # fit does nothing.

X2_test = transformer.transform(X2_test)

preds3 = clf2.predict(X2_test) # use predict_proba or a wrong prediction will result in infinity

print(preds3)

#clf.score(preds3, y2_test)

y_prob3 = clf2.predict_proba(X2_test)

print(y_prob3)
# LogisticRegression(((X_train, y_train), alpha=0.002), C=1.0, dual=True, penalty=l2)

# all parameters not specified are set to their defaults

from sklearn.linear_model import LogisticRegression

#logisticRegr = LogisticRegression()

logisticRegr = LogisticRegression(C=1.0, class_weight=None, dual=True, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='warn',

          n_jobs=None, penalty='l2', random_state=None, solver='warn',

          tol=0.0001, verbose=0, warm_start=False)



#logisticRegr.fit(((X_train, y_train), alpha=0.002), C=1.0, dual=True, penalty=l2)

logisticRegr.fit(X_train, y_train)

#logisticRegr = LogisticRegression(C=1.0, dual=True, penalty=l2)

#clf = LogisticRegression(random_state=0, solver='lbfgs',

#...                          multi_class='multinomial').fit(X, y)





predictions = logisticRegr.predict(X_test)

predictions_probs = logisticRegr.predict_proba(X_test)

print(predictions_probs)

# Use score method to get accuracy of model

score = logisticRegr.score(X_test, y_test)

print(score)
pred_proba_df = pd.DataFrame(data=preds, columns=['win2', 'win1']) # win1 is target

pred_proba_df['baseline'] = .5



print(f"The log loss of this simple model is {log_loss(y_test, pred_proba_df['win1']):.4f} while the baseline"

      f" (.5 for every team) is {log_loss(y_test, pred_proba_df['baseline']):.4f}.")
# source: https://stackoverflow.com/questions/14745022/how-to-split-a-column-into-two-columns

sub['Season'], sub['Team1'], sub['Team2'] = sub['ID'].str.split('_').str

sub[['Season', 'Team1', 'Team2']] = sub[['Season', 'Team1', 'Team2']].apply(pd.to_numeric)



sub_ = add_features(sub)

sub_X = sub_[['seed_numTeam1', 'seed_letterTeam1', 'seed_numTeam2', 'seed_letterTeam2']]



xgb = XGBClassifier()

xgb.fit(X,y)

sub_preds = xgb.predict_proba(sub_X)

print(sub_preds)

pred_proba_full_df = pd.DataFrame(data=sub_preds, columns=['win2', 'win1']) # win1 is target

sub['Pred'] = pred_proba_full_df['win1']

sub[['ID', 'Pred']].to_csv('submission.csv', index=False)



#### Next do this again for the logistic regression model submission file 



sub2 = sub

sub2['Season'], sub2['Team1'], sub2['Team2'] = sub2['ID'].str.split('_').str

sub2[['Season', 'Team1', 'Team2']] = sub2[['Season', 'Team1', 'Team2']].apply(pd.to_numeric)

sub_2 = add_features(sub2)

sub_X2 = sub_2[['seed_numTeam1', 'seed_letterTeam1', 'seed_numTeam2', 'seed_letterTeam2']]

sub_X2.head()



sub_preds_2 = logisticRegr.predict_proba(sub_X2)

print(sub_preds_2)



pred_proba_full_df_2 = pd.DataFrame(data=sub_preds_2, columns=['win2', 'win1']) # win1 is target

sub2['Pred'] = pred_proba_full_df_2['win1']

sub2[['ID', 'Pred']].to_csv('submission_LR.csv', index=False)



print(sub2)







tt =  pd.read_csv("../input/daily-rpi-2019/SampSubmishStage1_Prepped.csv")

tt.head()

features2 = ['PPG_1', 'OPPG_1', 'TOPG_1', 'FG3_PCT_1', 'AEE_1', 'PPG_2', 'OPPG_2', 'TOPG_2', 'FG3_PCT_2', 'AEE_2', 'seed_diff']

X2_tt = tt[features2]



transformer = Normalizer().fit(X2_tt) # fit does nothing.

X2_tt = transformer.transform(X2_tt)

prds = clf2.predict(X2_tt) # use predict_proba or a wrong prediction will result in infinity

print(prds)

#clf.score(preds3, y2_test)

y_prb3 = clf2.predict_proba(X2_tt)

print(y_prb3)

pred_proba_full_df_3 = pd.DataFrame(data=y_prb3, columns=['win2','win1']) #win1 is target

tt['Pred'] = pred_proba_full_df_3['win1']

tt[['Pred']].to_csv('submission_SVM_draft.csv', index=False)









#pipeline_optimizer.fit(X_train, y_train)

#print(pipeline_optimizer.score(X_test, y_test))

#pipeline_optimizer.export('tpot_exported_pipeline.py')