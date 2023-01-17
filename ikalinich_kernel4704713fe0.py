# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#learn
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import gc
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_V2.csv').set_index(['matchId', 'groupId'])
print(train.memory_usage())
train.head()
train.isnull().sum()
import gc
gc.collect()
%matplotlib inline
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap=sns.color_palette("RdBu", 20))

ax.set_xlabel('Numeric variables', size=14, color="#3498DB")
ax.set_ylabel('Numeric variables', size=14, color="#3498DB")
ax.set_title('[Heatmap] Correlation Matrix', size=18, color="#3498DB")

plt.show()
def generateFeatures(df):
    #df['speed'] = (df['rideDistance'] + df['walkDistance'] + df['swimDistance']) / df['matchDuration']
    #df['accuracy'] = df['headshotKills'] / df['kills']
    #df['items'] = df['boosts'] + df['heals']
    
    df = df[[
    'boosts',
    'heals',
    'damageDealt',
    'DBNOs',
    'headshotKills',
    'kills',
    'matchDuration',
    'revives',
    'rideDistance',
    'walkDistance',
    'swimDistance'
]]
    df_size = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    print('size')
    df_mean = df.groupby(['matchId','groupId']).mean().reset_index(['matchId','groupId'])
    print('mean')
    df_sum = df.groupby(['matchId','groupId']).sum().reset_index(['matchId','groupId'])
    print('sum')
    #df_max = df.groupby(['matchId','groupId']).max().reset_index(['matchId','groupId'])
    #print('max')
    #df_min = df.groupby(['matchId','groupId']).min().reset_index(['matchId','groupId'])
    #print('min')
    df_match_mean = df.groupby(['matchId']).mean().reset_index(['matchId'])
    print('matchMean')
    
    df = pd.merge(df, df_size, how='left', on=['matchId', 'groupId'])
    del df_size
    print('df_size')
    df = pd.merge(df, df_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
    del df_mean
    print('df_mean')
    df = pd.merge(df, df_sum, suffixes=["", "_sum"], how='left', on=['matchId', 'groupId'])
    del df_sum
    print('df_sum')
    #df = pd.merge(df, df_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
    #del df_max
    #print('df_max')
    #df = pd.merge(df, df_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
    #del df_min
    #print('df_min')
    df = pd.merge(df, df_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    del df_match_mean
    print('df_match_mean')
        
    #columns = list(df.columns)
    #columns.remove("matchId")
    #columns.remove("groupId")
    
    #df = df[columns]
    return df.fillna(0).set_index(['matchId', 'groupId'])
trainFeatures = generateFeatures(train)
print('Group metrics calculated')
trainFeatures.head()
trainFeatures['winPlacePerc'] = train['winPlacePerc']
print('Apend win metrics')
#trainFeatures = trainFeatures[~trainFeatures.index.duplicated(keep='first')]
#trainFeatures = trainFeatures.drop_duplicates(['matchId', 'groupId']).drop(['matchId', 'groupId'], axis=1)
print('cleanup groups')

trainFeatures.head(30)
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(trainFeatures.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap=sns.color_palette("RdBu", 20))

ax.set_xlabel('Numeric variables', size=14, color="#3498DB")
ax.set_ylabel('Numeric variables', size=14, color="#3498DB")
ax.set_title('[Heatmap] Correlation Matrix', size=18, color="#3498DB")

plt.show()
trainFeatures = trainFeatures.dropna().sample(frac=1)[:100000]
y = trainFeatures.pop('winPlacePerc').values
X = trainFeatures.values
X.shape
regressors = [
    svm.SVR(gamma='scale'),
    linear_model.SGDRegressor(tol=1e-10),
    linear_model.BayesianRidge(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(n_jobs=4, n_estimators=10)
]
for regressor in regressors:
    score = cross_val_score(regressor, X, y, cv=5).mean()
    print(f'{type(regressor).__name__} -- {score}')
regressor = RandomForestRegressor(n_jobs=4, n_estimators=10)
regressor.fit(X, y)
test = pd.read_csv('../input/test_V2.csv').set_index(['matchId', 'groupId'])
testFeatures = generateFeatures(test)
test.shape
test_X = testFeatures.values
test_X.shape
predictions = regressor.predict(test_X).reshape(-1,1)
sub = pd.DataFrame(predictions, index=test['Id']).rename(columns={0:'winPlacePerc'})
sub.head()
sub.to_csv('submission.csv', header=True)