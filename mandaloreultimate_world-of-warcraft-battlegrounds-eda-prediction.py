import numpy as np

import pandas as pd



import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg' 



import warnings

warnings.filterwarnings("ignore")
def my_confusion_matrix(y_test, y_pred, title):

    from sklearn.metrics import confusion_matrix, accuracy_score

    cm_df = pd.DataFrame(data=confusion_matrix(y_test, y_pred),

                         index=['Alliance', 'Horde'], 

                         columns=['Alliance', 'Horde'])

    plt.figure(figsize=(5.5,4))

    sns.heatmap(cm_df, annot=True, cmap='Blues')

    plt.title(title + '\nAccuracy:{:1.1f}%'.format(accuracy_score(y_test, y_pred) * 100))

    plt.ylabel('True winner')

    plt.xlabel('Predicted winner')

    plt.show()

    

def my_bar_plot(ax, name, data, x, y, xlabel='', ylabel='', orientation='horizontal',

                palette='icefire', format_spec='{:1.2f}%'):

    ax.set_title(name)

    sns.barplot(x=x, y=y, data=data, ax=ax, palette=palette)

    ax.set(xlabel=xlabel, ylabel=ylabel)

    

    if orientation == 'horizontal':

        for p in ax.patches:

            text = p.get_width()

            ax.text(x=p.get_x() + p.get_width() / 2., 

                y=p.get_y() + p.get_height() * 0.75,

                s=format_spec.format(text),

                ha="center",

                size="small",

                color='white')

        for tick in ax.get_yticklabels():

            tick.set_color(class_colormap[tick.get_text()])

    

    elif orientation == 'vertical':

        for p in ax.patches:

            text = p.get_height()

            ax.text(x=p.get_x() + p.get_width() / 2., 

                y=p.get_y() + p.get_height() / 2.,

                s=format_spec.format(text),

                ha="center",

                size="small",

                color='white')

        for tick in ax.get_xticklabels():

            tick.set_color(class_colormap[tick.get_text()])

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
import os

print('%-32s %d' % ('Input files available:', len(os.listdir('../input'))))

for i in range(32 + len(str(len(os.listdir('../input'))))):

    print('-',end='')

print('-')

for file in os.listdir("../input/"):

    unit = 'MB'

    size = os.stat('../input/' + file).st_size

    size = round(size / 1024, 2)

    unit = 'KB'

    print('%-25s %6.2f %2s' % (file, size, unit))
wowbgs = pd.read_csv('../input/wowbgs2.csv')

wowgil = pd.read_csv('../input/wowgil2.csv')

wowtk = pd.read_csv('../input/wowtk2.csv')

wowsm = pd.read_csv('../input/wowsm2.csv')

wowwg = pd.read_csv('../input/wowwg2.csv')
wowbgs.fillna(0, inplace=True)

wowbgs.drop(['Lose'], axis=1, inplace=True)

wowbgs.rename(columns={'Rol': 'Role'}, inplace=True)



wowgil.fillna(0, inplace=True)

wowgil.drop(['Lose'], axis=1, inplace=True)

wowbgs.rename(columns={'Rol': 'Role'}, inplace=True)



wowtk.fillna(0, inplace=True)

wowtk.drop(['Lose'], axis=1, inplace=True)

wowbgs.rename(columns={'Rol': 'Role'}, inplace=True)



wowsm.fillna(0, inplace=True)

wowsm.drop(['Lose'], axis=1, inplace=True)

wowbgs.rename(columns={'Rol': 'Role'}, inplace=True)



wowwg.fillna(0, inplace=True)

wowwg.drop(['Lose'], axis=1, inplace=True)

wowbgs.rename(columns={'Rol': 'Role'}, inplace=True)



bgs_dict = {'AB': 'Arathi Basin',

            'BG': 'Battle for Gilneas',

            'DG': 'Deepwind Gorge',

            'ES': 'Eye of the Storm',

            'SA': 'Strand of the Ancients',

            'SM': 'Silvershard Mines',

            'SS': 'Seething Shore',

            'TK': 'Temple of Kotmogu',

            'TP': 'Twin Peaks',

            'WG': 'Warsong Gulch'}

wowbgs['Battleground'].replace(bgs_dict, inplace=True)

class_names = sorted(wowbgs['Class'].unique())

matches_num = len(wowbgs['Code'].unique())



print('Dataframe shape:', wowbgs.shape)

print('Information on', matches_num, 'matches available.')

wowbgs.head()
faction_class_mix = wowbgs.pivot_table(values='Honor',

                                       index='Faction',

                                       columns='Class',

                                       aggfunc=lambda x: x.value_counts().count()).astype(int)

faction_class_mix.rename({'Death Knight': 'Death\nKnight', 'Demon Hunter': 'Demon\nHunter'}, axis=1, inplace=True)

_, ax = plt.subplots(1, 1, figsize=(16.5, 2.25))

sns.heatmap(faction_class_mix, annot=True, cmap='Blues', fmt='g', ax=ax)

plt.title('Faction/Class Comparison')

ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

ax.set_yticklabels(ax.get_yticklabels(), va='center')

ax.set(ylabel='', xlabel='')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))

ax.set_title("Roles Popularity")

sns.countplot(x='Role',

              data=wowbgs,

              ax=ax,

              palette='icefire_r',

              order = wowbgs['Role'].value_counts().index)

ax.set(xlabel='', ylabel='Frequency')



#Adding percentage to the patches

total = float(len(wowbgs))

for p in ax.patches:

    width = p.get_height()

    ax.text(x=p.get_x() + p.get_width()/2., 

              y=p.get_y() + p.get_height()*0.45,

              s='{:1.2f}%'.format(width/total* 100),

              ha="center",

              size="small",

              color='white')   

plt.show()
heal_classes = sorted(list(wowbgs[wowbgs['Role'] == 'heal']['Class'].unique()))

dps_classes = sorted(list(wowbgs[wowbgs['Role'] == 'dps']['Class'].unique()))



print('Healer classes:', *(heal_classes), sep='\n')

print('-' * 12)

print('DPS classes:', *(dps_classes), sep='\n')
_, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))

ax.set_title("Classes Popularity")

sns.countplot(y='Class',

              data=wowbgs,

              ax=ax,

              palette='icefire',

              order = wowbgs['Class'].value_counts().index)

ax.set(ylabel='', xlabel='Frequency')



#Adding percentage to the patches

total = float(len(wowbgs))

for p in ax.patches:

    width = p.get_width()

    ax.text(x=p.get_x() + p.get_width() / 2., 

              y=p.get_y() + p.get_height() * 0.75,

              s='{:1.2f}%'.format(width/total * 100),

              ha="center",

              size="small",

              color='white')



#Classes names color map

heal_dict, dps_dict = {x: 'darkblue' for x in heal_classes}, {x: 'darkred' for x in dps_classes}

class_colormap = {**dps_dict, **heal_dict}



for tick in ax.get_yticklabels():

    tick.set_color(class_colormap[tick.get_text()])

plt.show()
class_winrate = round(wowbgs.groupby(['Class'], as_index=False)['Win'].mean().sort_values(by=['Win'], ascending=False), 4)

class_winrate['Win'] *= 100



_, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))

my_bar_plot(ax, name="Classes Win Rate", data=class_winrate, x='Win',

            y='Class', orientation='horizontal', format_spec='{:1.2f}%')

plt.show()
_, ax = plt.subplots(1, 1, figsize=(8, 2.5))

ax.set_title("Heal/DPS Classes")

sns.countplot(x='Class',

              hue='Role',

              data=wowbgs[wowbgs['Class'].isin(heal_classes)],

              ax=ax,

              palette='icefire',

              order=wowbgs[wowbgs['Class'].isin(heal_classes)]['Class'].value_counts().index)

ax.set(ylabel='Frequency', xlabel='')



dps_patches = ax.patches[:5]

heal_patches = ax.patches[5:]

patches = list(zip(dps_patches, heal_patches ))



for p in patches:

    height = [p[i].get_height() for i in range(2)]

    total = sum(height)

    for i in range(2):

        ax.text(x=p[i].get_x() + p[i].get_width() / 2., 

              y=p[i].get_y() + p[i].get_height() * 0.45,

              s='{:1.1f}%'.format(height[i] / total * 100),

              ha="center",

              size="small",

              color='white')

    

plt.show()
mean_dd = round(wowbgs.groupby(['Class'], as_index=False)['DD'].mean().sort_values(by=['DD'], ascending=False))



_, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))

my_bar_plot(ax, name="Average Damage Dealt/Match by Classes", data=mean_dd, x='DD',

            y='Class', orientation='horizontal', format_spec='{:1.0f}')

plt.show()
median_kb = round(wowbgs.groupby(['Class'], as_index=False)['KB'].median().sort_values(by=['KB'], ascending=False))

median_hk = round(wowbgs.groupby(['Class'], as_index=False)['HK'].median().sort_values(by=['HK'], ascending=False))



_, axs = plt.subplots(1, 2, figsize=(11, 2.5))

my_bar_plot(axs[0], name="Average Kills/Match by Class", data=median_kb, x='Class',

            y='KB', ylabel='Median Kills', orientation='vertical', format_spec='{:1.0f}')

my_bar_plot(axs[1], name="Average Assists/Match by Class", data=median_hk, x='Class',

            y='HK', ylabel='Median Assists', orientation='vertical', format_spec='{:1.0f}')

plt.show()
median_d = round(wowbgs.groupby(['Class'], as_index=False)['D'].median().sort_values(by=['D'], ascending=False))



_, ax = plt.subplots(1, 1, figsize=(5.5, 2.5))

my_bar_plot(ax, name="Average Deaths/Match by Class", data=median_d, x='Class',

            y='D', ylabel='Median Deaths', orientation='vertical', format_spec='{:1.0f}')

plt.show()
mean_hd = round(wowbgs.groupby(['Class'], as_index=False)['HD'].mean().sort_values(by=['HD'], ascending=False))



_, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))

my_bar_plot(ax, name="Average Healing Done/Match by Classes", data=mean_hd, x='HD',

            y='Class', orientation='horizontal', format_spec='{:1.0f}')

plt.show()
from scipy.stats import norm

_, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))

ax.set_title("Damage Dealt/Match Distribution")

sns.distplot(wowbgs['DD'],

             fit=norm,

             ax=ax,

             kde_kws={'label': 'KDE'},

             fit_kws={'label': 'Normalized'})

ax.set(ylabel='', xlabel='')

plt.legend()

plt.show()
_, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))

ax.set_title("Healing Done/Match Distribution")

sns.distplot(wowbgs['HD'],

             fit=norm,

             ax=ax,

             kde_kws={'label': 'KDE'},

             fit_kws={'label': 'Normalized'})

ax.set(ylabel='', xlabel='')

plt.legend()

plt.show()
corr = wowbgs.corr(method='pearson')



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



_, ax = plt.subplots(figsize=(10,7))

ax.set_title("Player Stats Correlation")

sns.heatmap(corr, mask=mask, vmax=1, center=0, annot=True, fmt='.1f',

            square=True, linewidths=.5, cbar_kws={"shrink": .5});

plt.show()
honor_df = wowbgs.groupby(['Honor'], as_index=False)['Win'].mean()

honor_df['Size'] = np.array(wowbgs.groupby(['Honor'])['Win'].count())



_, ax = plt.subplots(1, 1, figsize=(7, 7))

ax.set_title("Honor/Win Rate Dependance")

sns.scatterplot(x="Honor", y="Win", data=honor_df, ax=ax, hue="Size", size="Size")

plt.show()
team_stats = ['DPS', 'Healers', 'Kills', 'Deaths', 'Assists', 'Damage', 'Healing']

wowbgs.rename(columns={'KB': 'Kills', 'D': 'Deaths', 'HK': 'Assists', 'DD': 'Damage', 'HD': 'Healing'}, inplace=True)

wowbgs['DPS'] = (wowbgs['Role'] == 'dps').astype(int)

wowbgs['Healers'] = (wowbgs['Role'] == 'heal').astype(int)

wowbgs.drop(['Role', 'Honor', 'BE'], axis=1, inplace=True)

wowbgs = pd.get_dummies(wowbgs, columns=['Class'])

for name in class_names:

    wowbgs.rename(columns={'Class_' + name: name}, inplace=True)



matches_columns = ['Battleground']

for faction in ['Alliance', 'Horde']:

    matches_columns += [faction + ' ' + name for name in class_names]

    matches_columns += [faction + ' ' + stat for stat in team_stats]



matches = pd.DataFrame(columns=matches_columns, index=range(matches_num))



matches['Battleground'] = np.array(wowbgs.groupby(['Code'])['Battleground'].first())

matches['Alliance Won'] = np.array(wowbgs[wowbgs['Faction'] == 'Alliance'].groupby(['Code'])['Win'].first().astype(int))

for faction in ['Alliance', 'Horde']:

    for stat in team_stats:

        matches[faction +' '+ stat] = np.array(wowbgs[wowbgs['Faction'] == faction].groupby(['Code'])[stat].sum())

    for name in class_names:

        matches[faction +' '+ name] = np.array(wowbgs[(wowbgs['Faction'] == faction)].groupby(['Code'])[name].sum())

matches.iloc[:,1:] = matches.iloc[:,1:].astype(int)



print('New dataset size:', matches.shape)

print('-'*27)

print('New features list:', *(matches.columns), sep='\n')

matches.head()
print('Is Alliance Kills number always equal to Horde Deaths number?',

      (matches['Alliance Kills'] == matches['Horde Deaths']).all())

print('Is Horde Kills number always equal to Alliance Deaths number?',

      (matches['Alliance Kills'] == matches['Horde Deaths']).all())
_, axs = plt.subplots(1, 2, figsize=(11, 5.5))

sns.scatterplot(ax=axs[0], x="Alliance Deaths", y="Horde Kills",

                hue="Alliance Won", data=matches, palette='icefire_r')

y = x = np.linspace(0, 95)

sns.lineplot(x=x, y=x, ax=axs[0])

sns.scatterplot(ax=axs[1], x="Horde Deaths", y="Alliance Kills",

                hue="Alliance Won", data=matches, palette="icefire_r", legend=False)

sns.lineplot(x=x, y=x, ax=axs[1])

plt.show()
_, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))

ax.set_title("Factions Win Rate")

sns.countplot(x='Alliance Won',

              data=matches,

              ax=ax,

              palette='icefire')

ax.set(xlabel='', ylabel='Matches Won')



total = float(len(matches))

for p in ax.patches:

    width = p.get_height()

    ax.text(x=p.get_x() + p.get_width() / 2., 

              y=p.get_y() + p.get_height() * 0.45,

              s='{:1.2f}%'.format(width/total * 100),

              ha="center",

              size="small",

              color='white')

plt.xticks(range(2), ('Alliance', 'Horde'))

plt.show()
bgs_matches = pd.DataFrame(columns=['Battleground, Matches'])

bgs_matches['Battleground'] = matches.groupby(['Battleground']).count().iloc[:,0].sort_values(ascending=False).index

bgs_matches['Matches'] = np.array(matches.groupby(['Battleground']).count().iloc[:,0].sort_values(ascending=False))



_, ax = plt.subplots(1, 1, figsize=(4.5, 3))

ax.set_title('Matches/Battleground')

sns.barplot(x='Matches',

            y='Battleground',

            data=bgs_matches,

            ax=ax,

            palette='icefire')

ax.set(xlabel='', ylabel='')

for p in ax.patches:

    width = p.get_width()

    ax.text(x=p.get_x() + p.get_width() / 2., 

    y=p.get_y() + p.get_height() * 0.75,

    s='{:.0f}'.format(width),

    ha="center",

    size="small",

    color='white')

plt.show()
class_columns = []

for faction in ['Alliance', 'Horde']:

    for name in class_names:

        class_columns += [faction + ' ' + name]

        

classes_corr = matches[class_columns]

corr = classes_corr.corr(method='pearson')



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



_, ax = plt.subplots(figsize=(12,11.5))

ax.set_title("Faction-Class Correlation")

cmap = sns.diverging_palette(220, 10, as_cmap = True )

sns.heatmap(corr, cmap = cmap, mask=mask, vmax=1, center=0,

            annot=True, fmt='.1f', square=True, linewidths=.5,

            cbar_kws={"shrink": .5});

plt.show()
matches['Battleground'] = pd.factorize(matches['Battleground'])[0]

np.random.seed(42)

matches = matches.sample(frac=1, random_state=42)

split = (0.7, 0.8)

rand_idx = np.random.randint(round(split[0] * (matches_num - 1)), round(split[1] * (matches_num - 1)))

df_train = matches[:rand_idx]

df_test = matches[rand_idx:]



print('Dataset divided')

print('Train sample size:', len(df_train), 'matches |', '{:d}%'.format(round(len(df_train) / len(matches) * 100)))

print('Train sample size:', len(df_test), 'matches  |', '{:d}%'.format(round(len(df_test) / len(matches) * 100)))
X_train = df_train.drop(['Alliance Won'], axis=1)

y_train = df_train['Alliance Won'].values

X_test = df_test.drop(['Alliance Won'], axis=1)

y_test = df_test['Alliance Won'].values
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV



xgb = XGBClassifier(tree_method='gpu_hist')



param_grid = {

    'n_estimators': [50, 100, 200, 500, 1000],

    'max_depth': [1, 3, 5, 7, 12, 15],

    'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],

    'gamma': [0, 0.5, 1, 5],

    'reg_alpha': [0.1, 0.25, 0.5],

    'reg_lambda': [0.1, 0.25, 0.5]

}



folds = 3

param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)
#Randomized Search log in hidden cell

random_search = RandomizedSearchCV(xgb, param_distributions=param_grid,

                                   n_iter=param_comb, scoring='roc_auc',

                                   n_jobs=-1, cv=skf.split(X_train,y_train),

                                   verbose=3, random_state=42)

random_search.fit(X_train, y_train)
print('XGB model best hyperparameters:')

print(random_search.best_params_)

print('XGB model best cross-validation score:')

print(random_search.best_score_)
y_pred = random_search.predict(X_test)

my_confusion_matrix(y_test, y_pred, 'XGB+RandomizedSearch')
from hyperopt import fmin, hp, tpe, Trials, space_eval

from sklearn.model_selection import KFold, cross_val_score



space={

       'n_estimators': hp.quniform('n_estimators', 1, 500, 50),

       'max_depth' : hp.quniform('max_depth', 2, 20, 1),

       'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.9),

       'reg_lambda': hp.uniform('reg_lambda', 0.1, 1.0),

       'learning_rate': hp.loguniform('learning_rate', 1e-4, 0.3),

       'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),

       'gamma': hp.uniform('gamma', 0.0, 5.0),

       'num_leaves': hp.choice('num_leaves', list(range(2, 15))),       

       'min_child_samples': hp.choice('min_child_samples', list(range(2, 10))),

       'feature_fraction': hp.choice('feature_fraction', [.5, .6, .7, .8, .9]),

       'bagging_fraction': hp.choice('bagging_fraction', [.5, .6, .7, .8, .9])

      }



# trials will contain logging information

trials = Trials()

num_folds=5

kf = KFold(n_splits=num_folds, random_state=42)
from sklearn.model_selection import cross_val_score

def xgb_cv(params, random_state=42, cv=kf, X=X_train, y=y_train):

    params = {

        'n_estimators': int(params['n_estimators']),

        'max_depth': int(params['max_depth']),

        'gamma': "{:.3f}".format(params['gamma']),

        'reg_alpha': "{:.3f}".format(params['reg_alpha']),

        'learning_rate': "{:.3f}".format(params['learning_rate']),

        'gamma': "{:.3f}".format(params['gamma']),

        'num_leaves': '{:.3f}'.format(params['num_leaves']),

        'min_child_samples': '{:.3f}'.format(params['min_child_samples']),

        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),

        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])

    }

    model = XGBClassifier(**params, tree_method='gpu_hist', random_state=42)

    score = -cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    return score
%%time

best=fmin(fn=xgb_cv,

          space=space, 

          algo=tpe.suggest,

          max_evals=35,

          trials=trials,

          rstate=np.random.RandomState(42)

         )

best['max_depth'] = int(best['max_depth'])

best['n_estimators'] = int(best['max_depth'])
#Hyperopt log in hidden cell

print(*(trials.results), sep='\n')
print('Best model parameters found with Hyperopt:\n', best)
model_2 = XGBClassifier(**best, tree_method='gpu_hist')

model_2.fit(X_train, y_train)

y_pred_2 = model_2.predict(X_test)

my_confusion_matrix(y_test, y_pred_2, 'XGB+Hyperopt')
feature_importances = pd.DataFrame(columns=['Feature', 'Importance'])

feature_importances['Feature'] = df_train.iloc[:,:-1].columns

feature_importances['Importance'] = np.array(model_2.feature_importances_)

feature_importances = feature_importances.sort_values(by=['Importance'], ascending=False).reset_index(drop=True)



_, ax = plt.subplots(1, 1, figsize=(9.5, 11))

ax.set_title('Feature Importances')

sns.barplot(x='Importance',

            y='Feature',

            data=feature_importances,

            ax=ax,

            palette='icefire')

ax.set(xlabel='', ylabel='')

for p in ax.patches:

    width = p.get_width()

    if width < 0.01:

        continue

    ax.text(x=p.get_x() + p.get_width() / 2., 

    y=p.get_y() + p.get_height() * 0.75,

    s='{:.2f}'.format(width),

    ha="center",

    size="small",

    color='white')

plt.show()
_, axs = plt.subplots(3, 2, figsize=(13, 19.5))



sns.scatterplot(ax=axs[0, 0], x="Alliance Kills", y="Horde Kills",

                hue="Alliance Won", data=matches, palette='icefire_r')

y = x = np.linspace(0, 95)

sns.lineplot(x=x, y=x, ax=axs[0, 0])



sns.scatterplot(ax=axs[0, 1], x="Alliance Assists", y="Horde Assists",

                hue="Alliance Won", data=matches, palette="icefire_r", legend=False)

y = x = np.linspace(0, 1200)

sns.lineplot(x=x, y=x, ax=axs[0, 1])



sns.scatterplot(ax=axs[1, 0], x="Alliance Damage", y="Horde Damage",

                hue="Alliance Won", data=matches, palette='icefire_r')

y = x = np.linspace(0, 1.5 * (10 ** 6))

sns.lineplot(x=x, y=x, ax=axs[1, 0])

xlabels = ['{:,.0f}'.format(x) + 'K' for x in axs[1, 0].get_xticks() / 1000]

ylabels = ['{:,.0f}'.format(x) + 'K' for x in axs[1, 0].get_yticks() / 1000]

axs[1, 0].set_xticklabels(xlabels)

axs[1, 0].set_yticklabels(ylabels)



sns.scatterplot(ax=axs[1, 1], x="Alliance Healing", y="Horde Healing",

                hue="Alliance Won", data=matches, palette="icefire_r", legend=False)

y = x = np.linspace(0, 1e6)

sns.lineplot(x=x, y=x, ax=axs[1, 1])

xlabels = ['{:,.0f}'.format(x) + 'K' for x in axs[1, 1].get_xticks() / 1000]

ylabels = ['{:,.0f}'.format(x) + 'K' for x in axs[1, 1].get_yticks() / 1000]

axs[1, 1].set_xticklabels(xlabels)

axs[1, 1].set_yticklabels(ylabels)



sns.scatterplot(ax=axs[2, 0], x="Alliance Damage", y="Horde Healing",

                hue="Alliance Won", data=matches, palette="icefire_r", legend=False)

y = x = np.linspace(0, 1.5 * 10**6)

sns.lineplot(x=x, y=0.7 * x, ax=axs[2, 0])

xlabels = ['{:,.0f}'.format(x) + 'K' for x in axs[1, 1].get_xticks() / 1000]

ylabels = ['{:,.0f}'.format(x) + 'K' for x in axs[1, 1].get_yticks() / 1000]

axs[2, 0].set_xticklabels(xlabels)

axs[2, 0].set_yticklabels(ylabels)



sns.scatterplot(ax=axs[2, 1], x="Horde Damage", y="Alliance Healing",

                hue="Alliance Won", data=matches, palette="icefire_r", legend=False)

y = x = np.linspace(0, 1.5 * 10**6)

sns.lineplot(x=x, y=0.7 * x - 0.5 * 10**5, ax=axs[2, 1])

xlabels = ['{:,.0f}'.format(x) + 'K' for x in axs[1, 1].get_xticks() / 1000]

ylabels = ['{:,.0f}'.format(x) + 'K' for x in axs[1, 1].get_yticks() / 1000]

axs[2, 1].set_xticklabels(xlabels)

axs[2, 1].set_yticklabels(ylabels)



plt.show()
import plotly.express as px



diff = pd.DataFrame(columns=['Kills Diff', 'Assists Diff', 'Damage Diff', 'Alliance Won'])

diff['Kills Diff'] = matches['Alliance Kills'] - matches['Horde Kills']

diff['Damage Diff'] = matches['Alliance Damage'] - matches['Horde Damage']

diff['Assists Diff'] = matches['Alliance Assists'] - matches['Horde Assists']

diff['Alliance Won'] = matches['Alliance Won']



fig = px.scatter_3d(diff, x='Kills Diff', y='Assists Diff', z='Damage Diff',

                    color='Alliance Won', opacity=0.75, color_continuous_scale='magma')

fig.show()
print('Matches Battlegrounds:', *(wowbgs['Battleground'].unique()), sep='\n')
gil_matches = wowgil.pivot_table(values=['BA', 'BD'], index='Code', columns=['Faction'], aggfunc=lambda x: x.sum())

gil_matches.columns = [(col[1] + ' ' + col[0]) for col in gil_matches.columns]

gil_matches['Alliance Won'] = np.array(wowgil[wowgil['Faction'] == 'Alliance'].groupby(['Code'])['Win'].first())



_, axs = plt.subplots(1, 2, figsize=(13, 6.5))



sns.scatterplot(ax=axs[0], x="Alliance BA", y="Horde BA",

                hue="Alliance Won", data=gil_matches, palette='icefire_r')

y = x = np.linspace(0, 8)

sns.lineplot(x=x, y=x, ax=axs[0])



sns.scatterplot(ax=axs[1], x="Alliance BD", y="Horde BD",

                hue="Alliance Won", data=gil_matches, palette="icefire_r", legend=False)

y = x = np.linspace(0, 4)

sns.lineplot(x=x, y=x, ax=axs[1])



plt.show()
sm_matches = wowsm.pivot_table(values='CC', index='Code', columns='Faction', aggfunc=lambda x: x.sum())

sm_matches['Alliance Won'] = wowsm[wowsm['Faction'] == 'Alliance'].groupby(['Code'])['Win'].first().astype(int)

sm_matches.rename({'Alliance': 'Alliance CC', 'Horde': 'Horde CC'}, axis=1, inplace=True)



_, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))



sns.scatterplot(ax=ax, x="Alliance CC", y="Horde CC",

                hue="Alliance Won", data=sm_matches, palette='icefire_r')

y = x = np.linspace(0, 45)

sns.lineplot(x=x, y=x, ax=ax)



plt.show()
tk_matches = wowtk.pivot_table(values=['OP', 'VP'], index='Code', columns=['Faction'], aggfunc=lambda x: x.sum())

tk_matches.columns = [(col[1] + ' ' + col[0]) for col in tk_matches.columns]

tk_matches['Alliance Won'] = np.array(wowtk[wowtk['Faction'] == 'Alliance'].groupby(['Code'])['Win'].first())



_, axs = plt.subplots(1, 2, figsize=(13, 6.5))



sns.scatterplot(ax=axs[0], x="Alliance OP", y="Horde OP",

                hue="Alliance Won", data=tk_matches, palette='icefire_r')

y = x = np.linspace(0, 25)

sns.lineplot(x=x, y=x, ax=axs[0])



sns.scatterplot(ax=axs[1], x="Alliance VP", y="Horde VP",

                hue="Alliance Won", data=tk_matches, palette="icefire_r", legend=False)

y = x = np.linspace(0, 1600)

sns.lineplot(x=x, y=x, ax=axs[1])



plt.show()
wg_matches = wowwg.pivot_table(values=['FC', 'FR'], index='Code', columns=['Faction'], aggfunc=lambda x: x.sum())

wg_matches.columns = [(col[1] + ' ' + col[0]) for col in wg_matches.columns]

wg_matches['Alliance Won'] = np.array(wowwg[wowwg['Faction'] == 'Alliance'].groupby(['Code'])['Win'].first())



_, axs = plt.subplots(1, 2, figsize=(13, 6.5))



sns.scatterplot(ax=axs[0], x="Alliance FC", y="Horde FC",

                hue="Alliance Won", data=wg_matches, palette='icefire_r')

y = x = np.linspace(0, 8)

sns.lineplot(x=x, y=x, ax=axs[0])



sns.scatterplot(ax=axs[1], x="Alliance FR", y="Horde FR",

                hue="Alliance Won", data=wg_matches, palette="icefire_r", legend=False)

y = x = np.linspace(0, 7)

sns.lineplot(x=x, y=x, ax=axs[1])



plt.show()