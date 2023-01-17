import numpy as np  

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/video-games-sales-prediction/Train.csv')

test = pd.read_csv('/kaggle/input/video-games-sales-prediction/Test.csv')

sub = pd.read_csv('/kaggle/input/video-games-sales-prediction/Sample_Submission.csv')
train.shape, test.shape, sub.shape
train.head(5)
train.isnull().sum()
train.nunique()
sub.head()
sns.distplot(train['SalesInMillions']);
import matplotlib.style as style

style.available

style.use('bmh')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

sns.distplot(train['CRITICS_POINTS'], ax=ax1)

sns.distplot(train['USER_POINTS'], ax=ax2);
plt.figure(figsize=(15,5))

sns.countplot(train['CONSOLE'], order=train['CONSOLE'].value_counts().index);
plt.figure(figsize=(15,5))

sns.countplot(train['CATEGORY'], order=train['CATEGORY'].value_counts().index);
plt.figure(figsize=(15,5))

sns.countplot(train['PUBLISHER'], order=train['PUBLISHER'].value_counts().iloc[:40].index)

plt.xticks(rotation=90);
plt.figure(figsize=(15,5))

sns.countplot(train['YEAR'])

plt.xticks(rotation=90);
plt.figure(figsize=(12, 8))



train_corr = train.corr()

sns.heatmap(train_corr, 

            xticklabels = train_corr.columns.values,

            yticklabels = train_corr.columns.values,

            annot = True);
sns.pairplot(train, hue='RATING', diag_kind='kde');
g = sns.catplot(x="CONSOLE", y="SalesInMillions", data=train);

g.fig.set_size_inches(15,5)
g = sns.catplot(x="CONSOLE", y="SalesInMillions", kind="box", data=train)

g.fig.set_size_inches(15,5);
g = sns.catplot(x="YEAR", y="SalesInMillions", data=train)

g.fig.set_size_inches(15,5);
g = sns.catplot(x="CATEGORY", y="SalesInMillions", data=train)

g.fig.set_size_inches(15,5);
g = sns.catplot(x="RATING", y="SalesInMillions", data=train)

g.fig.set_size_inches(15,5);
train['RATING'].value_counts()
test['RATING'].value_counts()
x = train.groupby(['CATEGORY']).sum().copy()

ax = x['SalesInMillions'].sort_values(ascending=False).plot(kind='bar', figsize=(13, 5));



for p in ax.patches:

    ax.annotate(str( round( p.get_height() ) ) + "\n" + str(round( p.get_height() /89.170) )+ "%", 

                (p.get_x() * 1.007, p.get_height() * 0.75),

                color='black')
sns.jointplot(x='CRITICS_POINTS',y='USER_POINTS',data=train, kind='hex', size=5);
train = train[train['RATING'] != 'RP']

train = train[train['RATING'] != 'K-A']
train = train[train['SalesInMillions'] < 30]
df = train.append(test,ignore_index=True)

df.shape
df.head(3)
agg_func = {

    'CRITICS_POINTS': ['mean','min','max','sum'],

    'USER_POINTS': ['mean','min','max','sum']

}

agg_func = df.groupby('CONSOLE').agg(agg_func)

agg_func.columns = [ 'CONSOLE_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['CONSOLE'], how='left')



agg_func = {

    'CRITICS_POINTS': ['mean','min','max','sum'],

    'USER_POINTS': ['mean','min','max','sum']

}

agg_func = df.groupby('CATEGORY').agg(agg_func)

agg_func.columns = [ 'CATEGORY_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['CATEGORY'], how='left')



agg_func = {

    'CRITICS_POINTS': ['mean','min','max','sum'],

    'USER_POINTS': ['mean','min','max','sum']

}

agg_func = df.groupby('PUBLISHER').agg(agg_func)

agg_func.columns = [ 'PUBLISHER_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['PUBLISHER'], how='left')



agg_func = {

    'CRITICS_POINTS': ['mean','min','max','sum'],

    'USER_POINTS': ['mean','min','max','sum']

}

agg_func = df.groupby('RATING').agg(agg_func)

agg_func.columns = [ 'RATING_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['RATING'], how='left')
#df['Old'] = 2020 - df['YEAR']
df['Unique_CATEGORY_per_CONSOLE'] = df.groupby(['CONSOLE'])['CATEGORY'].transform('nunique')
calc = df.groupby(['CONSOLE'], axis=0).agg({'CONSOLE':[('op1', 'count')]}).reset_index() 

calc.columns = ['CONSOLE','CONSOLE Count']

df = df.merge(calc, on=['CONSOLE'], how='left')



calc = df.groupby(['CATEGORY'], axis=0).agg({'CATEGORY':[('op1', 'count')]}).reset_index() 

calc.columns = ['CATEGORY','CATEGORY Count']

df = df.merge(calc, on=['CATEGORY'], how='left')



calc = df.groupby(['PUBLISHER'], axis=0).agg({'PUBLISHER':[('op1', 'count')]}).reset_index() 

calc.columns = ['PUBLISHER','PUBLISHER Count']

df = df.merge(calc, on=['PUBLISHER'], how='left')



calc = df.groupby(['RATING'], axis=0).agg({'RATING':[('op1', 'count')]}).reset_index() 

calc.columns = ['RATING','RATING Count']

df = df.merge(calc, on=['RATING'], how='left')
for c in ['ID', 'CONSOLE', 'CATEGORY', 'PUBLISHER', 'RATING']:

    df[c] = df[c].astype('category')
#df.drop('ID', axis=1, inplace=True)
agg_func = {

    'CRITICS_POINTS': ['mean','sum']   

}

agg_func = df.groupby(['YEAR','CONSOLE']).agg(agg_func)

agg_func.columns = [ 'YEAR_CONSOLE_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['YEAR','CONSOLE'], how='left')
# 7th Gen: ps3, x360, wii

# 8th Gen: ps4, xone, wiiu



def check_if_latest(console):

    if console in ['ps3','x360','wii','ps4','xone','wiiu']:

        return 1

    else:

        return 0    



df['LATEST'] = df['CONSOLE'].apply(check_if_latest)
df['Count_ID_per_PUBLISHER'] = df.groupby(['PUBLISHER'])['ID'].transform('count')
train_df = df[df['SalesInMillions'].isnull()!=True]

test_df = df[df['SalesInMillions'].isnull()==True]

test_df.drop(['SalesInMillions'], axis=1, inplace=True)
X = train_df.drop(labels=['SalesInMillions'], axis=1)

y = train_df['SalesInMillions'].values



X.shape, y.shape
X.head(3)
from math import sqrt 

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor
Xtest = test_df
errlgb = []

y_pred_totlgb = []



fold = KFold(n_splits=15, shuffle=True, random_state=2019)



for train_index, test_index in fold.split(X):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    param = {'objective': 'regression',

         'boosting': 'gbdt',  

         'metric': 'l2_root',

         'learning_rate': 0.1, 

         'num_iterations': 2500,

         'num_leaves': 20,

         'max_depth': -1,

         'min_data_in_leaf': 15,

         'bagging_fraction': 0.9,

         'bagging_freq': 1,

         'feature_fraction': 0.9,

         'early_stopping_round': 100,

         }



    lgbm = LGBMRegressor(**param)

    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=100)



    y_pred_lgbm = lgbm.predict(X_test)

    print("RMSE LGBM: ", sqrt(mean_squared_error(y_test, y_pred_lgbm)))



    errlgb.append(sqrt(mean_squared_error(y_test, y_pred_lgbm)))

    p = lgbm.predict(Xtest)

    y_pred_totlgb.append(p)



print('\nMean RMSE', np.mean(errlgb,0))
feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importances_, X.columns), reverse=True)[:], 

                           columns=['Value','Feature'])

plt.figure(figsize=(15, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()
final_lgb = np.mean(y_pred_totlgb,0)
sub['SalesInMillions'] = final_lgb
sub.head()
sub.to_csv('Submission.csv', index=False)