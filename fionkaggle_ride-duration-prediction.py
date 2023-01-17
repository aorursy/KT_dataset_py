%matplotlib inline

import pandas as pd

from datetime import datetime

import pandas as pd

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge

from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import mean_squared_error

from math import radians, cos, sin, asin, sqrt

import seaborn as sns

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]
test= pd.read_csv('../input/Test.csv')

train=pd.read_csv('../input/Train (2).csv')

pd.set_option('display.float_format', lambda x: '%.3f' % x)

test.head()
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train.head()
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train.describe()
plt.hist(train['Time from Pickup to Arrival'].values, bins=100)

plt.xlabel('Time from Pickup to Arrival')

plt.ylabel('number of train records')

plt.show()


m = np.mean(train['Time from Pickup to Arrival'])

s = np.std(train['Time from Pickup to Arrival'])

#train = train[train['Time from Pickup to Arrival'] <= m + 2*s]

#train2 = train[train['Time from Pickup to Arrival'] >= m - 2*s]

train2 = train[train['Time from Pickup to Arrival'] >= 180]

pd.set_option('display.float_format', lambda x: '%.3f' % x)

train2.describe()
train2.info()
plt.hist(train2['Time from Pickup to Arrival'].values, bins=100)

plt.xlabel('Time from Pickup to Arrival')

plt.ylabel('number of train records')

plt.show()
train['log_Time from Pickup to Arrival'] = np.log(train['Time from Pickup to Arrival'].values + 1)

plt.hist(train['log_Time from Pickup to Arrival'].values, bins=100)

plt.xlabel('log(Time from Pickup to Arrival)')

plt.ylabel('number of train records')

plt.show()

sns.distplot(train["log_Time from Pickup to Arrival"], bins =100)
plt.plot(train2.groupby('Pickup - Day of Month').count()[['Distance (KM)']], 'o-', label='train')

plt.plot(test.groupby('Pickup - Day of Month').count()[['Distance (KM)']], 'o-', label='test')

plt.title('Trips over Time.')

plt.legend(loc=0)

plt.ylabel('Trips')

plt.show()

plt.plot(train2.groupby('Temperature').mean()[['Time from Pickup to Arrival']], 'o-', label='train')



plt.title('Trips over Time.')

plt.legend(loc=0)

plt.ylabel('Trips')

plt.show()
pc = train2.groupby('Pickup - Day of Month')['Time from Pickup to Arrival'].mean()

plt.title('Trip duration and Day of Month')

plt.ylabel('Time in Seconds')

sns.barplot(pc.index,pc.values)
#city_long_border = (-74.03, -73.75)

#city_lat_border = (40.63, 40.85)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)

ax[0].scatter(train2['Pickup Long'].values[:100000], train2['Pickup Lat'].values[:100000],

              color='blue', s=1, label='train', alpha=0.1)

ax[1].scatter(test['Pickup Long'].values[:100000], test['Pickup Lat'].values[:100000],

              color='green', s=1, label='test', alpha=0.1)

fig.suptitle('Train and test area overlap.')

ax[0].legend(loc=0)

ax[0].set_ylabel('latitude')

ax[0].set_xlabel('longitude')

ax[1].set_xlabel('longitude')

ax[1].legend(loc=0)

#plt.ylim(city_lat_border)

#plt.xlim(city_long_border)

plt.show()
vehicle_train = pd.get_dummies(train2['Vehicle Type'], prefix='veh', prefix_sep='_')

vehicle_test = pd.get_dummies(test['Vehicle Type'], prefix='veh', prefix_sep='_')

personal_train = pd.get_dummies(train2['Personal or Business'], prefix='per', prefix_sep='_')

personal_test = pd.get_dummies(test['Personal or Business'], prefix='per', prefix_sep='_')
vehicle_train.shape,vehicle_test.shape
personal_train.shape,personal_test.shape
train2 = train2.drop(['Order No','User Id','Vehicle Type','Personal or Business','Placement - Time','Confirmation - Time','Arrival at Pickup - Time','Pickup - Time','Arrival at Destination - Time','Rider Id','Precipitation in millimeters'],axis=1)
test = test.drop(['Order No','User Id','Vehicle Type','Personal or Business','Placement - Time','Confirmation - Time','Arrival at Pickup - Time','Pickup - Time','Rider Id','Precipitation in millimeters'],axis=1)
train2 = train2.drop(['Arrival at Destination - Day of Month','Arrival at Destination - Weekday (Mo = 1)'],axis=1)
train2.shape,test.shape
Train2= pd.concat([vehicle_train,personal_train,train2],axis=1)
Test2= pd.concat([vehicle_test,personal_test,test],axis=1)
Train2.shape,Test2.shape
# Since we have test data, I am not going to split the data

#train3, test3 = train_test_split(train2, test_size = 0.2)
y=(Train2['Time from Pickup to Arrival']/60) #change from seconds to minutes

Train2.shape,y.shape,Test2.shape
#Train2.info(),Test2.info()
cols_with_missing = [col for col in Train2.columns 

                                 if Train2[col].isnull().any()]

reduced_Train2 = Train2.drop(cols_with_missing, axis=1)

reduced_Test2 = Test2.drop(cols_with_missing, axis=1)

reduced_Train2.info(),reduced_Test2.info()
reduced_Train2 = reduced_Train2.drop(['Time from Pickup to Arrival'],axis=1)

reduced_Train2.info(),reduced_Test2.info()
from sklearn.linear_model import LinearRegression,BayesianRidge,ElasticNet,Lasso,SGDRegressor,Ridge

lasso = Lasso(alpha = 0.01)

lasso.fit(reduced_Train2,y)

y_pred_lasso = lasso.predict(reduced_Test2)
FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=reduced_Train2.columns)

FI_lasso.sort_values("Feature Importance",ascending=False)
FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(10,15))

plt.xticks(rotation=90)

plt.show()
from sklearn.decomposition import PCA,KernelPCA

pca = PCA(0.95)



#pca = PCA(n_components = 426)

PCA_reduced_Train2 = pca.fit_transform(reduced_Train2)

#X_scaled = pca.inverse_transform(lower_dimension_pca)

var1 = np.round(pca.explained_variance_ratio_*100, decimals = 1)

var1
label =['PC' + str(x) for x in range(1,len(var1)+1)]

plt.figure(figsize=(15,12))

plt.bar(x=range(1,len(var1)+1), height = var1 ,tick_label = label)



plt.ylabel("Percentage of Explained Variance")

plt.xlabel("Principle Components")

plt.title("Principle Component Analysis")

plt.show()
from sklearn.model_selection import cross_val_score,KFold,GridSearchCV,RandomizedSearchCV,StratifiedKFold,train_test_split



# Define Root Mean Square Error 

def rmse_cv(model,X,y):

    rmse = np.sqrt(-cross_val_score(model,X,y,scoring="neg_mean_squared_error",cv=5))

    return rmse
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor,VotingClassifier

from sklearn.svm import LinearSVR,SVR

from sklearn.kernel_ridge import KernelRidge

from xgboost import XGBRegressor



models = [LinearRegression(),

             Ridge(),

             Lasso(alpha=0.01,max_iter=10000),

             RandomForestRegressor(),

             GradientBoostingRegressor(),

             SVR(),

             LinearSVR(),

             ElasticNet(alpha = 0.001,max_iter=10000),

             SGDRegressor(max_iter=1000, tol = 1e-3),

             BayesianRidge(),

             KernelRidge(alpha=0.6,kernel='polynomial',degree = 2,coef0=2.5),

             ExtraTreesRegressor(),

             XGBRegressor()]



names = ['LR','Ridge','Lasso','RF','GBR','SVR','LSVR','ENet','SGDR','BayRidge','Kernel','XTreeR','XGBR']
for model,name in zip(models,names):

    score = rmse_cv(model,reduced_Train2,y)

    print("{}: {:.6f}, {:4f}".format(name,score.mean(),score.std()))
class grid():

    def __init__(self,model):

        self.model = model

    def grid_get(self,X,y,param_grid):

        grid_search = GridSearchCV(self.model,param_grid,cv=5,scoring='neg_mean_squared_error')

        grid_search.fit(X,y)

        print(grid_search.best_params_,np.sqrt(-grid_search.best_score_))

        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])

        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
grid(Lasso()).grid_get(reduced_Train2,y,{'alpha':[0.01,0.001,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0009],

                                       'max_iter':[10000]})
grid(Ridge()).grid_get(reduced_Train2,y,

                       {'alpha':[10,20,25,30,35,40,45,50,55,57,60,65,70,75,80,100],'max_iter':[10000]})