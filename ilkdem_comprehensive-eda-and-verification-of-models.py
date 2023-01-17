import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy import stats 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,learning_curve
from sklearn.metrics import mean_absolute_error,make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
df_train = pd.read_csv(r'../input/train.csv')
df_test = pd.read_csv(r'../input/test.csv')
df = pd.concat([df_train,df_test],axis = 0)
print("Train set shape : ",df_train.shape," Test set shape : ",df_test.shape)
print("Train set columns : ",sorted(df_train.columns))
print("Test set columns : ",sorted(df_test.columns))
df.info()
df.datetime = pd.to_datetime(df.datetime)
df.describe()
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df['dayofweek'] = df['datetime'].dt.dayofweek
df['day'] = df['datetime'].dt.day

binaryVariables = ['holiday','workingday']
categoricVariables = ['season','weather']
numericVariables = ['atemp','humidity','windspeed']
dummySeason = pd.get_dummies(df.season,drop_first = True,prefix = 'season')
dummyWeather = pd.get_dummies(df.weather,drop_first = True,prefix = 'weather')
df = pd.concat([df,dummySeason,dummyWeather],axis = 1)
df_trainCorrected = df[df['count'].isnull() == False]
df_testCorrected = df[df['count'].isnull() == True]
plt.figure(figsize = (20,5))
plt.subplot(131)
sns.distplot(df_trainCorrected['count'])
plt.subplot(132)
sns.distplot(df_trainCorrected['registered'])
plt.subplot(133)
sns.distplot(df_trainCorrected['casual'])
plt.show()
plt.figure(figsize = (20,10))
plt.subplot(221)
sns.distplot(df_trainCorrected['count'])
plt.subplot(222)
stats.probplot(df_trainCorrected['count'],plot = plt)
plt.subplot(223)
sns.distplot(np.log(df_trainCorrected['count']))
plt.subplot(224)
stats.probplot(np.log(df_trainCorrected['count']),plot = plt)
plt.show()
df_trainCorrected[df_trainCorrected['count'] != df_trainCorrected['registered'] + df_trainCorrected['casual']].shape
df_trainCorrected['rateCasual'] = df_trainCorrected['casual'] / df_trainCorrected['count']
plt.figure(figsize=(20,5))
plt.subplot(121)
sns.pointplot('month','count',data = df_trainCorrected,join = True,color = 'g')
sns.pointplot('month','registered',data = df_trainCorrected,join = True,color = 'r')
sns.pointplot('month','casual',data = df_trainCorrected,join = True,color = 'b')
plt.subplot(122)
sns.barplot('month','rateCasual',data = df_trainCorrected)
plt.show()
plt.figure(figsize=(20,10))
plt.subplot(221)
sns.boxplot('season','count',data = df_trainCorrected)
plt.subplot(222)
sns.boxplot('weather','count',data = df_trainCorrected)
plt.subplot(223)
sns.boxplot('workingday','count',data = df_trainCorrected)
plt.subplot(224)
sns.boxplot('holiday','count',data = df_trainCorrected)
plt.show()
plt.figure(figsize=(20,10))
plt.subplot(221)
sns.distplot(df_trainCorrected['atemp'])
plt.subplot(222)
sns.distplot(df_trainCorrected['temp'])
plt.subplot(223)
sns.distplot(df_trainCorrected['humidity'])
plt.subplot(224)
sns.distplot(df_trainCorrected['windspeed'])
plt.show()
plt.figure(figsize=(20,10))
plt.subplot(221)
sns.scatterplot('atemp','count',data = df_trainCorrected)
plt.subplot(222)
sns.scatterplot('temp','count',data = df_trainCorrected)
plt.subplot(223)
sns.scatterplot('humidity','count',data = df_trainCorrected)
plt.subplot(224)
sns.scatterplot('windspeed','count',data = df_trainCorrected)
plt.show()
df_trainCorrected[df_trainCorrected['humidity'] == 0].loc[:,['temp','humidity','datetime']]
avgPrevDay = df_trainCorrected[(df_trainCorrected['year'] == 2011) & (df_trainCorrected['month'] == 3) & (df_trainCorrected['day'] == 9)]['humidity'].mean()
avgNextDay = df_trainCorrected[(df_trainCorrected['year'] == 2011) & (df_trainCorrected['month'] == 3) & (df_trainCorrected['day'] == 11)]['humidity'].mean()
avgHumidity = (avgPrevDay + avgNextDay) / 2
df_trainCorrected.iloc[df_trainCorrected[df_trainCorrected['humidity'] == 0].index,:]['humidity'] = round(avgHumidity,0)
df_trainCorrected.loc[df_trainCorrected['windspeed'] == 0,'windspeed'] = np.nan
df_trainCorrected['windspeed'] = df_trainCorrected['windspeed'].fillna(df_trainCorrected.groupby(['year','month','day'])['windspeed'].transform("mean"))
print("Number of days which are holiday but working day : ",df_trainCorrected[(df_trainCorrected['holiday'] == 1) & (df_trainCorrected['workingday'] == 1)]['count'].sum())
print("Not holiday, not working day : ",df_trainCorrected[(df_trainCorrected['holiday'] == 0) & (df_trainCorrected['workingday'] == 0)]['dayofweek'].unique())
print("Working days : ",df_trainCorrected[df_trainCorrected['workingday'] == True]['dayofweek'].unique())
print("Holidays : ",df_trainCorrected[df_trainCorrected['holiday'] == True]['dayofweek'].unique())
plt.figure(figsize = (20,20))
plt.subplot(411)
sns.pointplot('hour','count',hue = 'season',data = df_trainCorrected)
plt.subplot(412)
sns.pointplot('hour','count',hue = 'dayofweek',data = df_trainCorrected)
plt.subplot(413)
sns.pointplot('dayofweek','count',hue = 'season',data = df_trainCorrected)
plt.subplot(414)
sns.pointplot('month','count',hue = 'year',data = df_trainCorrected)
plt.show()
plt.figure(figsize=(40,40))
sns.heatmap(df_trainCorrected.corr(),annot = True,square=True,fmt = '.2f')
plt.show()
df_trainCorrected['count'] = np.log1p(df_trainCorrected['count'])
def rmsle(y_test,y_pred):
    square_logarithmic_error = (y_test - y_pred) ** 2
    return np.sqrt(square_logarithmic_error.mean())
def printScores(y_test,y_pred):
    print("RMSLE : ",rmsle(y_test,y_pred))
    print("MAE : ",mean_absolute_error(y_test,y_pred))
dropColumns_train = ['atemp','casual','registered', 'count', 'datetime','rateCasual','season','weather']
X = df_trainCorrected.drop(dropColumns_train,axis = 1)
y = df_trainCorrected['count']
X_train, X_test, y_train, y_test = train_test_split(X,y)
param_grid = {
    "lasso__alpha":[0.1,1,5,10,100]
}
pipe = Pipeline([("scaler",StandardScaler()),("lasso",Lasso())])
rmsleScorer = make_scorer(rmsle,greater_is_better = False)
grid = GridSearchCV(pipe,param_grid = param_grid,scoring = rmsleScorer,cv = 5)
grid.fit(X_train,y_train)
y_pred_lasso = grid.best_estimator_.predict(X_test)
printScores(y_test,y_pred_lasso)
print(grid.best_params_)
train_sizes, train_scores, test_scores = learning_curve(grid.best_estimator_,X_train, y_train, cv = 5,scoring = rmsleScorer, train_sizes = range(1,int(len(X_train) * 0.75),500))
plt.plot(train_sizes,np.mean(train_scores,axis = 1),color = 'r',label = 'Train')
plt.plot(train_sizes,np.mean(test_scores,axis = 1),color = 'b',label = 'Test')
plt.legend(loc = 'best')
plt.show()
plt.figure(figsize = (20,5))
plt.bar(X_train.columns,grid.best_estimator_.named_steps['lasso'].coef_)
plt.xticks(rotation = 90)
plt.show()
param_grid = {
    "dtr__min_samples_split":[10,50,100],
    "dtr__max_depth" : [5,10,50,100]
}
pipe = Pipeline([("scaler",StandardScaler()),("dtr",DecisionTreeRegressor())])
rmsleScorer = make_scorer(rmsle,greater_is_better = False)
grid = GridSearchCV(pipe,param_grid = param_grid,scoring = rmsleScorer,cv = 5)
grid.fit(X_train,y_train)
y_pred_dtr = grid.best_estimator_.predict(X_test)
printScores(y_test,y_pred_dtr)
print(grid.best_params_)
importances = grid.best_estimator_.named_steps['dtr'].feature_importances_

for i,j in zip(importances,X_train.columns):
    if i > 0.01:
        print(j,round(i,2))
columns = ['humidity','temp','workingday','hour','month','year','dayofweek','weather_3']
param_grid = {
    "rfr__max_depth" : [10,20,50,100],
    "rfr__n_estimators" : [10,50,100],
    'rfr__max_features' : [3, 4, 5, 6]
}
pipe = Pipeline([("scaler",StandardScaler()),("rfr",RandomForestRegressor())])
grid = GridSearchCV(pipe,param_grid = param_grid,scoring = rmsleScorer,cv = 5)
grid.fit(X_train[columns],y_train)
y_pred_rfr = grid.best_estimator_.predict(X_test[columns])
printScores(y_pred_rfr,y_test)
print(grid.best_params_)
train_sizes, train_scores, test_scores = learning_curve(grid.best_estimator_,X_train[columns], y_train, cv = 5,scoring = rmsleScorer, train_sizes = range(1,int(len(X_train) * 0.75),500))
plt.plot(train_sizes,np.mean(train_scores,axis = 1),color = 'r',label = 'Train')
plt.plot(train_sizes,np.mean(test_scores,axis = 1),color = 'b',label = 'Test')
plt.legend(loc = 'best')
plt.show()
param_grid = {
    "gbr__max_depth" : [10,20,50],
    "gbr__n_estimators" : [10,50,100],
    'gbr__learning_rate' : [0.001,0.1,1]
}
pipe = Pipeline([("scaler",StandardScaler()),("gbr",GradientBoostingRegressor())])
grid = GridSearchCV(pipe,param_grid = param_grid,scoring = rmsleScorer,cv = 5)
grid.fit(X_train[columns],y_train)
y_pred_gbr = grid.best_estimator_.predict(X_test[columns])
printScores(y_pred_gbr,y_test)
print(grid.best_params_)
train_sizes, train_scores, test_scores = learning_curve(grid.best_estimator_,X_train[columns], y_train, cv = 5,scoring = rmsleScorer, train_sizes = range(1,int(len(X_train) * 0.75),500))
plt.plot(train_sizes,np.mean(train_scores,axis = 1),color = 'r',label = 'Train')
plt.plot(train_sizes,np.mean(test_scores,axis = 1),color = 'b',label = 'Test')
plt.legend(loc = 'best')
plt.show()