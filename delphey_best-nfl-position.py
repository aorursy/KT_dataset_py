import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd 

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt



import lightgbm as lgb
df=pd.read_csv('../input/nfl-combine-data/combine_data_since_2000_PROCESSED_2018-04-26.csv')
df['Ht'] = df['Ht']/12

df['Ht'] = round(df['Ht'], 2)
df.head()
df['Pos'].unique()
df['Pos'] = df['Pos'].replace({'G': 'OG', 'OL': 'OG', 'S': 'FS', 'DB': 'CB', 'EDGE': 'DE', 'P': 'P/K', 'K': 'P/K', 'LB': 'ILB', 'MLB': 'ILB'})

df = df.loc[df['Pos'] != 'LS']
df.columns
df.shape
combinedata = df[['Pos','Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps',

       'BroadJump', 'Cone', 'Shuttle', 'Year']]



combinedata[['Ht', 'Forty', 'Vertical', 

       'BroadJump', 'Cone', 'Shuttle']] = combinedata[['Ht', 'Forty', 'Vertical', 

       'BroadJump', 'Cone', 'Shuttle']].astype(float)



combinedata[['Wt', 'BenchReps','Year']] = combinedata[['Wt', 'BenchReps','Year']].astype('Int32')



combinedata['Pos'] = combinedata['Pos'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(df, df['Pos'], test_size = 0.3, random_state = 42)
clf = lgb.LGBMClassifier()

clf.fit(X_train.drop(['Pos', 'Player', 'Pfr_ID', 'AV', 'Team', 'Round',

       'Pick'], axis=1), y_train, verbose=10)
y_pred=clf.predict(X_test.drop(['Pos', 'Player', 'Pfr_ID', 'AV', 'Team', 'Round',

       'Pick'], axis=1))

accuracy=accuracy_score(y_pred, y_test)

print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
X_train['train_test'] = 'train'

X_test['train_test'] = 'test'



alldata = X_train.append(X_test).sort_values('Year')
merow = np.array([5.85, 175, 5.1, 24, 2,

       90, np.nan, np.nan, 2020]).reshape(1, -1)



me = pd.DataFrame(columns=['Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps',

       'BroadJump', 'Cone', 'Shuttle', 'Year'], data=merow)



y_me=clf.predict(me)



y_me
alldata['pred'] = clf.predict(alldata.drop(['Pos', 'Player', 'Pfr_ID', 'AV', 'Team', 'Round',

       'Pick', 'train_test'], axis=1))
alldata.head(20)
X_train['Pos'] = X_train['Pos'].astype('category')

X_test['Pos'] = X_test['Pos'].astype('category')
X_train.columns
hyper_params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'mape',

    'learning_rate': 0.1,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.7,

    'bagging_freq': 20,

    'verbose': 0,

    "max_depth": 8,

    #"num_leaves": 128,  

    #"max_bin": 512,

    #"num_iterations": 100000,

    "n_estimators": 1000

}
gbm = lgb.LGBMRegressor(**hyper_params)
X_train['Pos'] = X_train['Pos'].astype('category')

X_test['Pos'] = X_test['Pos'].astype('category')
gbm.fit(X_train.loc[X_train['Pick']>0].drop(['Player', 'Pfr_ID', 'AV', 'Team', 'Round', 'Year', 'train_test', 'Pick'], axis=1), X_train.loc[X_train['Pick']>0]['Pick'],

        eval_set=[(X_test.loc[X_test['Pick']>0].drop(['Player', 'Pfr_ID', 'AV', 'Team', 'Round', 'Year', 'train_test', 'Pick'], axis=1), X_test.loc[X_test['Pick']>0]['Pick'])],

        verbose=10,

        early_stopping_rounds=10)
alldata['Pos'] = alldata['Pos'].astype('category')

alldata['predpick'] = gbm.predict(alldata.drop(['Player', 'Pick', 'Pfr_ID', 'AV', 'Team', 'Round',

        'Year', 'train_test', 'pred'], axis=1))
alldata.loc[alldata['Pick']>0].sort_values('predpick').head(20)
OL = ['C', 'OT', 'OG']

DL = ['NT', 'DT', 'DE']

LB = ['MLB', 'ILB', 'OLB']

DB = ['CB', 'SS', 'FS']



alldata['Pos Group'] = alldata['Pos'].astype(str)

alldata.loc[alldata['Pos Group'].isin(OL), 'Pos Group'] = 'OL'

alldata.loc[alldata['Pos Group'].isin(DL), 'Pos Group'] = 'DL'

alldata.loc[alldata['Pos Group'].isin(LB), 'Pos Group'] = 'LB'

alldata.loc[alldata['Pos Group'].isin(DB), 'Pos Group'] = 'DB'


sns.pairplot(alldata, x_vars=['Pick', 'predpick'], y_vars=['Pick', 'predpick'], hue='Pos Group', height=8,

            kind='reg')
sns.pairplot(alldata, x_vars=['Forty', 'BenchReps', 'Ht', 'Wt'], y_vars=['Forty', 'BenchReps', 'Ht', 'Wt'], hue='Pos Group', height=6,

            plot_kws={'alpha': 0.5,"s": 100})
sns.lineplot(data=alldata, x='Year', y='Forty')

ax2 = plt.twinx()

plt.xticks(np.arange(min(alldata['Year']), max(alldata['Year'])+1, 1))

sns.lineplot(data=alldata, x='Year', y='Wt', ax=ax2, color='orange')

g = sns.FacetGrid(col="Pos", data=alldata,

           col_wrap=5, height=5)

g.map(sns.lineplot, 'Year', 'Forty', color='blue')

plt.show()
g = sns.FacetGrid(col="Pos", data=alldata,

           col_wrap=5, height=5)

g.map(sns.lineplot, 'Year', 'Wt', color='orange')

plt.show()
sns.lineplot(data=alldata, x='Round', y='Forty')

ax2 = plt.twinx()

plt.xticks(np.arange(min(alldata['Round']), max(alldata['Round'])+1, 1))

sns.lineplot(data=alldata, x='Round', y='BenchReps', ax=ax2)
g = sns.catplot(x="Pos Group", y="Forty", hue="Round",

            kind="bar", data=alldata, height=8, aspect=2)

g.ax.set_ylim(4.2, 5.5)
g = sns.catplot(x="Pos Group", y="BenchReps", hue="Round",

            kind="bar", data=alldata, height=8, aspect=2)

#g.ax.set_ylim(4.2, 5.5)
g = sns.catplot(x="Pos", y="BenchReps", 

            kind="bar", data=alldata, height=8, aspect=2)

#g.ax.set_ylim(4.2, 5.5)
alldata.to_csv("alldata.csv")