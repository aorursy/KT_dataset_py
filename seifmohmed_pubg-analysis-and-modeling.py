# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import the important libraries
import pandas as pd     # for dataframe
import numpy as np      # for arraies
import matplotlib.pyplot as plt  # for visualization 
%matplotlib inline
import seaborn as sns           # for visualization 
# read the data
df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
df.head()
df.iloc[:,:14].head(10)
df.iloc[:,14:].head(10)
df.info(null_counts=True)
# for knowing statistical information
df.describe().T
df.winPlacePerc.value_counts().head()
df.winPlacePerc.plot(kind='hist',bins=100)
plt.title('number of winer in each place')
plt.show()
df.weaponsAcquired.value_counts().head()
df.nunique()
# how many duplicats in the data
df.duplicated().sum()
# firstly let's drop the nan in the winPlacePerc feture 
df.dropna(axis=0,inplace=True)
df.info(null_counts=1)
df.matchType.value_counts()
df.matchType.value_counts()
plt.figure(figsize=(15,6))
plt.xticks(rotation=45)
ax = sns.barplot(df.matchType.value_counts().index, df.matchType.value_counts().values, alpha=0.8)
ax.set_title("Number of players in the same match type")
plt.show()
df.matchType.replace(['squad-fpp','squad','normal-squad-fpp','normal-squad'],'squad',inplace=True)
df.matchType.replace(['duo-fpp','normal-duo-fpp','normal-duo'],'duo',inplace=True)
df.matchType.replace(['solo-fpp','normal-solo-fpp','normal-solo'],'solo',inplace=True)
df.matchType.replace(['crashfpp','flaretpp','flarefpp','crashtpp'],'others',inplace=True)
df.matchType.value_counts()
sns.barplot(df.matchType.value_counts().index, df.matchType.value_counts().values)
plt.title('squad vs duo vs solo')
plt.show()
plt.figure(figsize=(30,10))
plt.xticks(rotation=90)
sns.countplot(df.DBNOs,hue=df.matchType)
plt.title("DBNOs/match types")
plt.show()
plt.figure(figsize=(30,6))
plt.xticks(rotation=90)
sns.countplot(df.revives,hue=df.matchType)
plt.title("revives/match types")
plt.show()
plt.figure(figsize=(30,10))
plt.xticks(rotation=90)
sns.countplot(df.kills,hue=df.matchType)
plt.title("kills/match types")
plt.show()
sns.scatterplot(df.kills,df.damageDealt)
plt.title('relation between kills and damage')
plt.show()
kills = df[['headshotKills','killPlace','killPoints','kills'
            ,'killStreaks','longestKill','roadKills','teamKills'
            ,'weaponsAcquired','winPlacePerc']]
kills.head()
kills.nunique()
kills_features = ['headshotKills','killPlace','killPoints','kills'
                  ,'killStreaks','longestKill','roadKills','teamKills','weaponsAcquired']
def scatter (feature_name):
        sns.scatterplot(x=feature_name,y='winPlacePerc',data=kills)
plt.figure(figsize=(15,10))
plt.subplot(3,3,1)
scatter(kills_features[0])
plt.subplot(3,3,2)
scatter(kills_features[1])
plt.subplot(3,3,3)
scatter(kills_features[2])
plt.subplot(3,3,4)
scatter(kills_features[3])
plt.subplot(3,3,5)
scatter(kills_features[4])
plt.subplot(3,3,6)
scatter(kills_features[5])
plt.subplot(3,3,7)
scatter(kills_features[6])
plt.subplot(3,3,8)
scatter(kills_features[7])
plt.subplot(3,3,9)
scatter(kills_features[8])
plt.show()
kills.corr()
plt.figure(figsize=(8,8))
sns.heatmap(kills.corr(), annot=True, linewidths=.5)
mobility=df[['rideDistance','roadKills','swimDistance','vehicleDestroys','walkDistance','winPlacePerc']]
mobility.head()
# Then you map to the grid
g = sns.PairGrid(mobility)
g.map(plt.scatter)
sns.heatmap(mobility.corr(),annot=True, linewidths=.5)
data_train = df.copy()
data_train.head()
plt.figure(figsize=(20,20))
sns.heatmap(data_train.corr(),annot=True, linewidths=.5)
f_data_train=data_train.drop(['Id','groupId','matchId','killPoints','matchDuration','maxPlace',
                 'numGroups','rankPoints','roadKills','teamKills'
                 ,'winPoints','killStreaks','longestKill','killPoints'],axis=1)
f_data_train.head()

f_data_train.info()
X= f_data_train.drop('winPlacePerc',axis=1)
X.head()
y= data_train[['winPlacePerc']]
y.head()
num_data = X.drop('matchType',axis=1)
num_data.shape
cat_data = f_data_train['matchType']
cat_data = pd.DataFrame(cat_data)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('std_scaler', StandardScaler())])

data_num_tr = num_pipeline.fit_transform(num_data)
data_num_tr= pd.DataFrame(data_num_tr,columns=num_data.columns)
data_num_tr.head()
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False)
data_cat_1hot = cat_encoder.fit_transform(cat_data)
data_cat_1hot
cat_encoder.categories_
from sklearn.compose import ColumnTransformer

num_attribs = list(num_data)
cat_attribs = ["matchType"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

data_prepared = full_pipeline.fit_transform(f_data_train)
data_prepared=pd.DataFrame(data_prepared)
data_prepared.drop(17,axis=1,inplace=True)
X = data_prepared.copy()
y.head()
import xgboost as xgb

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)
param = {
    'eta': 0.15, 
    'max_depth': 5,  
    'num_class': 2} 

steps = 20  # The number of training iterations
model = xgb.train(param, D_train, steps)
from sklearn.metrics import mean_squared_error

preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("MSE = {}".format(mean_squared_error(Y_test, best_preds)))
test_data = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
test_data.head()
list(f_data_train.columns)
test_set = test_data[['assists','boosts','damageDealt','DBNOs','headshotKills',
 'heals','killPlace','kills','matchType','revives','rideDistance',
 'swimDistance','vehicleDestroys','walkDistance','weaponsAcquired']]
test_set.matchType.replace(['squad-fpp','squad','normal-squad-fpp','normal-squad'],'squad',inplace=True)
test_set.matchType.replace(['duo-fpp','normal-duo-fpp','normal-duo'],'duo',inplace=True)
test_set.matchType.replace(['solo-fpp','normal-solo-fpp','normal-solo'],'solo',inplace=True)
test_set.matchType.replace(['crashfpp','flaretpp','flarefpp','crashtpp'],'others',inplace=True)
test_set.head()
test_model = full_pipeline.fit_transform(test_set)
test_model
test_model = pd.DataFrame(test_model)
test_model.head()
test_model.drop(17,axis=1,inplace=True)
# X is dependant features
# y independant label
# test model for prediction
X.head()
y.head()
test_model.head()
X.to_csv('independant_feture_train.csv')
y.to_csv('dependant_feture_train.csv')
test_model.to_csv('test_model.csv')
X = pd.read_csv('independant_feture_train.csv',)
y = pd.read_csv('dependant_feture_train.csv')
test_model = pd.read_csv('test_model.csv')
X.drop('Unnamed: 0',axis=1,inplace=True)
y.drop('Unnamed: 0',axis=1,inplace=True)
test_model.drop('Unnamed: 0',axis=1,inplace=True)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error
forest_predictions = forest_reg.predict(X_test)
forest_mae = mean_absolute_error(y_test, forest_predictions)
forest_mae

'''
for tuning parameters
parameters_for_testing = {
    'colsample_bytree':[0.4,0.6,0.8],
    'gamma':[0,0.03,0.1,0.3],
    'min_child_weight':[1.5,6,10],
    'learning_rate':[0.1,0.07],
    'max_depth':[3,5],
    'n_estimators':[10000],
    'reg_alpha':[1e-5, 1e-2,  0.75],
    'reg_lambda':[1e-5, 1e-2, 0.45],
    'subsample':[0.6,0.95]  
}

                   
xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')
gsearch1.fit(train_x,train_y)
print (gsearch1.grid_scores_)
print('best params')
print (gsearch1.best_params_)
print('best score')
print (gsearch1.best_score_)
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
import xgboost 

xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
xgb_model.fit(X_train,y_train)
xgb_pred = xgb_model.predict(X_test)
mean_absolute_error(y_test, xgb_pred)
test_pred = xgb_model.predict(test_model)
submit = pd.read_csv('sample_submission_V2.csv')
submit.winPlacePerc = test_pred
submit.to_csv('xgboost_prediction.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,)
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=10000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X_train, y_train)
sgd_pred=sgd_reg.predict(X_test)
sqd_mae = mean_absolute_error(y_test,sgd_pred)
sqd_mae
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
lm = LinearRegression()
lm.fit(X_train,y_train)
lm_pred = lm.predict(X_test)
mean_absolute_error(y_test,lm_pred)
