# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pylab

import calendar

import numpy as np

import pandas as pd

import seaborn as sn

from scipy import stats

import missingno as msno

from datetime import datetime

import matplotlib.pyplot as plt

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
train = pd.read_csv('/kaggle/input/bike-share-demand/train.csv')



test = pd.read_csv('/kaggle/input/bike-share-demand/test.csv')

data = train.append(test)

data.reset_index(inplace=True)

data.drop('index',inplace=True,axis=1)
data.head()
data["date"] = data.datetime.apply(lambda x : x.split()[0])

data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])

#data["day"] = data.datetime.apply(lambda x : x.split()[0].split("-")[2])

data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())

data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)
data.head()
from sklearn.ensemble import RandomForestRegressor



dataWind0 = data[data["windspeed"]==0]

dataWindNot0 = data[data["windspeed"]!=0]

rfModel_wind = RandomForestRegressor()

windColumns = ["season","weather","humidity","month","temp","year","atemp"]

rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])



wind = rfModel_wind.predict(dataWind0[windColumns])



wind = pd.Series(wind)



a = 0

windnew = []

for i in data.loc[: , 'windspeed']:

    if i == 0:

        windnew.append(wind[a])

        a += 1

    else:

        windnew.append(i)

        

windnew = pd.Series(windnew)



data['windnew'] = windnew.values



data.drop(columns='windspeed', inplace=True)

data.head()
categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]

numericalFeatureNames = ["temp","humidity","windnew","atemp"]

dropFeatures = ['casual',"cnt","datetime","date"]

data_num = data[numericalFeatureNames]
# Plot skew value for each numerical value

from scipy.stats import skew 

skewness = data_num.apply(lambda x: skew(x))

skewness.sort_values(ascending=False)
fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sn.distplot(windnew,ax=ax1,bins=50)

sn.distplot(windnew,ax=ax2,bins=50)

plt.xlabel('windspeed')


from scipy import stats

skewness = skewness[abs(skewness) > 0.5]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

print("Mean skewnees: {}".format(np.mean(skewness)))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    data_num[feat] = boxcox1p(data_num[feat], stats.boxcox_normmax(data_num[feat] + 1))

    data[feat] = boxcox1p(data[feat], stats.boxcox_normmax(data[feat] + 1))

    

    

from scipy.stats import skew 

skewness.sort_values(ascending=False)





skewness = data_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]



print("There are {} skewed numerical features after Box Cox transform".format(skewness.shape[0]))

print("Mean skewnees: {}".format(np.mean(skewness)))

skewness.sort_values(ascending=False)
# Plot skew value for each numerical value

from scipy.stats import skew 

skewness = data_num.apply(lambda x: skew(x))

skewness.sort_values(ascending=False)
#     # seperating season as per values. this is bcoz this will enhance features.

# season=pd.get_dummies(data['season'],prefix='season')

# data=pd.concat([data,season],axis=1)

# data.head()





#        # # seperating season as per values. this is bcoz this will enhance features.

# season=pd.get_dummies(data['workingday'],prefix='workingday')

# data=pd.concat([data,season],axis=1)

# data.head()



# season=pd.get_dummies(data['year'],prefix='year')

# data=pd.concat([data,season],axis=1)

# data.head()



#     # # seperating season as per values. this is bcoz this will enhance features.

# season=pd.get_dummies(data['weather'],prefix='weather')

# data=pd.concat([data,season],axis=1)

# data.head()





#     # # seperating season as per values. this is bcoz this will enhance features.

# season=pd.get_dummies(data['hour'],prefix='hour')

# data=pd.concat([data,season],axis=1)

# data.head()





# season=pd.get_dummies(data['weekday'],prefix='weekday')

# data=pd.concat([data,season],axis=1)

# data.head()





# season=pd.get_dummies(data['month'],prefix='month')

# data=pd.concat([data,season],axis=1)

# data.head()





# season=pd.get_dummies(data['holiday'],prefix='holiday')

# data=pd.concat([data,season],axis=1)

# data.head()
## craete binning data continue 

add_contu = ['humidity', 'temp','atemp', 'windnew'

            ]

for i in range(len(add_contu)):

    x = add_contu[i]

    q1 = data[x].describe()[4]

    q2 = data[x].describe()[5]

    q3 = data[x].describe()[6]

    data.loc[data[x] <= q1 , x+'_bin' ] = 0

    data.loc[(data[x] > q1 ) & (data[x] <= q2 ), x+'_bin' ] = 1

    data.loc[(data[x] > q2 ) & (data[x] <= q3 ), x+'_bin' ] = 2

    data.loc[(data[x] > q3 ), x+'_bin'] = 3
data.columns
multyply = ['holiday', 'humidity', 'season', 'temp', 'weather', 

            'weekday', 'windnew',

            'workingday', 'hour', 'month']

data['year'] = data['year'].astype("int")
for i in range (0,len(multyply)):

    for j in range (i+1, len(multyply)):

        if i != j :

            x = multyply[i]

            y = multyply[j]

            data[x+y] = data[x] * data[y]

            
for x in multyply:

    data[x+'-s2'] = data[x] ** 2

    data[x+'-s3'] = data[x] ** 3

    data[x+'-sqr'] = np.sqrt(data[x])
data.columns
import seaborn as sns

data.cov()

sns.heatmap(data.corr())

plt.show()
data.shape
# from collections import OrderedDict

# from itertools import islice

# o = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 4)])

# sliced = islice(o.items(), 2)  # o.iteritems() is o.items() in Python 3

# sliced_o = OrderedDict(sliced)

# sliced_o


'''



Index([atemp, holiday, humidity, season, weather, weekday,

       'workingday', hour, year, windnew, holidayweekday,

       'holidayworkingday', 'humiditytemp', 'humiditywindnew', 'seasonweather',

       'seasonweekday', 'seasonworkingday', 'seasonhour', 'tempweather',

       'weatherhour', weekdayworkingday, weekdayhour', workingdayhour],

      dtype='object')

'''
# Create correlation matrix

corr_matrix = data.corr().abs()

# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
len(to_drop)
data[to_drop].cov()

sns.heatmap(data[to_drop].corr())

plt.show()
data = data.drop(to_drop, axis =1 )
data.shape
data.cov()

sns.heatmap(data.corr())

plt.show()
corrMatt = data.corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)



sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
data.shape
categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]

numericalFeatureNames = ["temp","humidity","windnew","atemp"]

dropFeatures = ['casual',"cnt","datetime","date"]



# spliting train test

dataTrain = data[pd.notnull(data['cnt'])].sort_values(by=["datetime"])

dataTest = data[~pd.notnull(data['cnt'])].sort_values(by=["datetime"])

datetimecol = dataTest["datetime"]

yLabels = train['cnt']

#yLablesRegistered = dataTrain["registered"]

yLablesCasual = dataTrain["casual"]
dataTrain.head()
dataTrain  = dataTrain.drop(dropFeatures,axis=1)

dataTest  = dataTest.drop(dropFeatures,axis=1)
def rmsle(actual, predict):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in actual]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in predict]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
print("Skewness: %f" % yLabels.skew())

print("Kurtosis: %f" % yLabels.kurt())
dataTrain.columns
# potongan = ['holiday','season','weather','weekday','workingday','hour','year']

# dataTrain.drop(columns = potongan, inplace=True)

# dataTest.drop(columns = potongan, inplace=True)
dataTrain.shape
import xgboost as xg

xgr=xg.XGBRegressor(max_depth=9,min_child_weight=1,gamma=0.0,colsample_bytree=0.6,subsample=0.6,

                   learning_rate = 0.01, n_estimators = 400)



yLabelsLog = np.log1p(yLabels)

xgr.fit(dataTrain,yLabelsLog)

from collections import OrderedDict

OrderedDict(sorted(xgr.get_booster().get_fscore().items(), key=lambda t: t[1], reverse=True))
# importantkolom = ['atemp','humidity','atemphumidity','atempwindnew','humiditywindnew','windnew','weekday_6',

#                  'weekday_5','weather_3','season_1','weekday_2','month_4','weekday_0','weekday_1','weekday_3']

# dataTrain = dataTrain.loc[:, importantkolom]

# dataTest = dataTest.loc[:, importantkolom]
# TESTING MODEL WITH TRAIN AND TEST DATA FROM dataTrain

from sklearn.model_selection import train_test_split

# create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(dataTrain, yLabels, test_size=0.3, random_state=42)
# from sklearn.ensemble import RandomForestRegressor

# rfModel = RandomForestRegressor(n_estimators=200)

# rfModel.fit(dataTrain,yLabelsLog)

# preds = rfModel.predict(X= dataTrain)

# print ("RMSLE Value For Random Forest: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
# from sklearn.ensemble import BaggingRegressor

# from sklearn.tree import DecisionTreeRegressor



# model = BaggingRegressor(base_estimator=DecisionTreeRegressor(),

#                                 max_features=1.0,

#                                 bootstrap_features=False,

#                                 random_state=42)

# model.fit(dataTrain,yLabelsLog)

# preds = model.predict(X= dataTrain)

# print ("RMSLE Value For BAGGING REGRESSOR: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
# from sklearn.ensemble import GradientBoostingRegressor

# gbm = GradientBoostingRegressor(learning_rate= 0.1, n_estimators=4000, max_depth=17, alpha= 0.1); ### Test 0.41



# #gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.1); ### Test 0.41

# yLabelsLog = np.log1p(yLabels)

# gbm.fit(dataTrain, yLabelsLog)

# preds = gbm.predict(X= dataTrain)

# print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(yLabelsLog),np.exp(preds)))
# TESTING MODEL WITH TRAIN AND TEST DATA FROM dataTrain

from sklearn.model_selection import train_test_split

# create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(dataTrain, yLabels, test_size=0.3, random_state=42)
import xgboost as xg

xgr=xg.XGBRegressor(max_depth=9,min_child_weight=1,gamma=0.0,colsample_bytree=0.6,subsample=0.6,

                   learning_rate = 0.01, n_estimators = 400, reg_lambda = 3)# BEST LR AND N ESTIMATOR {'learning_rate': 0.0001, 'n_estimators': 100}



# from sklearn.ensemble import GradientBoostingRegressor

# gbm = GradientBoostingRegressor(learning_rate= 0.1, n_estimators=4000, max_depth=17, alpha= 0.1); ### Test 0.41



#gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.1); ### Test 0.41

yLabelsLog = np.log1p(y_train)

xgr.fit(X_train,yLabelsLog)

preds = xgr.predict(X_test)

def rmsle(actual, predict):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in actual]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in predict]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))

print ("RMSLE Value For Gradient Boost: ",rmsle(y_test,np.exp(preds)))
0.315

0.311
import xgboost as xg

xgr=xg.XGBRegressor(max_depth=9,min_child_weight=1,gamma=0.0,colsample_bytree=0.6,subsample=0.6,

                   learning_rate = 0.01, n_estimators = 400, reg_lambda =3)# BEST LR AND N ESTIMATOR {'learning_rate': 0.0001, 'n_estimators': 100}

#maxdep 3, minchild 5, 0.4



# from sklearn.ensemble import GradientBoostingRegressor

# gbm = GradientBoostingRegressor(learning_rate= 0.1, n_estimators=4000, max_depth=17, alpha= 0.1); ### Test 0.41



#gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.1); ### Test 0.41

yLabelsLog = np.log1p(y_train)

xgr.fit(X_train,yLabelsLog)



from sklearn.model_selection import KFold # import KFold

kf = KFold(n_splits=5, random_state=None, shuffle=False)

kf.get_n_splits(X_train)



def rmsle(actual, predict):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in actual]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in predict]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))

from sklearn.metrics import fbeta_score, make_scorer

from sklearn.model_selection import cross_val_score



rmsle = make_scorer(rmsle)

scores = cross_val_score(xgr, X_test, y_test, cv=kf, scoring= rmsle)

print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))
0.383

0.374
import xgboost as xg

xgr=xg.XGBRegressor(max_depth=3,min_child_weight=5,gamma=0.4,colsample_bytree=0.6,subsample=0.6,

                   learning_rate = 0.0001, n_estimators = 100)# BEST LR AND N ESTIMATOR {'learning_rate': 0.0001, 'n_estimators': 100}

#maxdep 3, minchild 5, 0.4



# from sklearn.ensemble import GradientBoostingRegressor

# gbm = GradientBoostingRegressor(learning_rate= 0.1, n_estimators=4000, max_depth=17, alpha= 0.1); ### Test 0.41



#gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.1); ### Test 0.41

yLabelsLog = np.log1p(yLabels)

xgr.fit(dataTrain,yLabelsLog)



from sklearn.model_selection import KFold # import KFold

kf = KFold(n_splits=5, random_state=None, shuffle=False)

kf.get_n_splits(dataTrain)



def rmsle(actual, predict):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in actual]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in predict]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))

from sklearn.metrics import fbeta_score, make_scorer

from sklearn.model_selection import cross_val_score



rmsle = make_scorer(rmsle)

scores = cross_val_score(xgr, dataTrain, yLabelsLog, cv=kf, scoring= rmsle)

print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))
# Loss: 0.098 (+/- 0.025) --> KETIKA KORELASI LEBIH DARI 0.9

# Loss: 0.098 (+/- 0.027) --> KETIKA KORELASIH LEBIH DARI 0.8
scores.mean()
fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sn.distplot(yLabels,ax=ax1,bins=50)

sn.distplot(yLabelsLog,ax=ax2,bins=50)

import xgboost as xg

xgr=xg.XGBRegressor(max_depth=9,min_child_weight=1,gamma=0.0,colsample_bytree=0.6,subsample=0.6,

                   learning_rate = 0.01, n_estimators = 400, reg_lambda =1.5)# BEST LR AND N ESTIMATOR {'learning_rate': 0.0001, 'n_estimators': 100}

#maxdep 3, minchild 5, 0.4



# from sklearn.ensemble import GradientBoostingRegressor

# gbm = GradientBoostingRegressor(learning_rate= 0.1, n_estimators=4000, max_depth=17, alpha= 0.1); ### Test 0.41



#gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.1); ### Test 0.41

yLabelsLog = np.log1p(yLabels)

xgr.fit(dataTrain,yLabelsLog)



predsTest = xgr.predict( dataTest)

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sn.distplot(yLabels,ax=ax1,bins=50)

sn.distplot(np.exp(predsTest),ax=ax2,bins=50)



# from sklearn.linear_model import Ridge,Lasso, ElasticNet

# from sklearn.ensemble import GradientBoostingRegressor

# import xgboost as xg

# from mlxtend.regressor import StackingCVRegressor

# gbm = GradientBoostingRegressor(learning_rate= 0.01, n_estimators=400, max_depth=17, alpha= 0.1)

# ridge = Ridge(alpha=0.1)

# lasso = Lasso(alpha=0.1)

# elasticnet = ElasticNet(random_state=0)

# xgr=xg.XGBRegressor(max_depth=9,min_child_weight=1,gamma=0.0,colsample_bytree=0.6,subsample=0.6,

#                    learning_rate = 0.01, n_estimators = 400)









# stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbm, xgr),

#                                 meta_regressor=xgr,

#                                 use_features_in_secondary=True)



# yLabelsLog = np.log1p(yLabels)

# stack_gen.fit(dataTrain,yLabelsLog)
predsTest
dataTrain.columns
dataTest.columns
# predsTest = stack_gen.predict( dataTest)
submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in np.exp(predsTest)]

    })
submission['count'] = round(submission['count'])
submission.head()
dataTrain.columns




from IPython.display import HTML

import pandas as pd

import numpy as np

submission.to_csv('submission.csv', index=False)



def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='submission.csv')
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

def rmsle(y, y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))

from sklearn.metrics import fbeta_score, make_scorer

rmsle = make_scorer(rmsle)



model = xg.XGBRegressor(max_depth=9,min_child_weight=1,gamma=0.0,colsample_bytree=0.6,subsample=0.6,

                   learning_rate = 0.01, n_estimators = 400)

learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

n_estimators = [100, 200, 300, 400, 500]

param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)

kf = KFold(n_splits=10, shuffle=True, random_state=7) # Define the split - into 2 folds 

grid_search = GridSearchCV(model, param_grid, scoring=rmsle, n_jobs=-1, cv=kf)

yLabelsLog = np.log1p(yLabels)

grid_result = grid_search.fit(dataTrain, yLabelsLog)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

	print("%f (%f) with: %r" % (mean, stdev, param))

from sklearn.model_selection import KFold

def rmsle(y, y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))

from sklearn.metrics import fbeta_score, make_scorer

rmsle = make_scorer(rmsle)



model = xg.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4,colsample_bytree=0.6,subsample=0.6)

param_test1 = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,6,2)

}

kf = KFold(n_splits=10, shuffle=True, random_state=7) # Define the split - into 2 folds

grid_search = GridSearchCV(model,param_grid = param_test1, scoring=rmsle,n_jobs=-1, cv=kf)

yLabelsLog = np.log1p(yLabels)



grid_result = grid_search.fit(dataTrain, yLabelsLog)

# summarize results



means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

	print("%f (%f) with: %r" % (mean, stdev, param))

print("Best: %f using %s" % ( grid_result.best_score_, grid_result.best_params_))
from sklearn.model_selection import KFold

def rmsle(y, y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))

from sklearn.metrics import fbeta_score, make_scorer

rmsle = make_scorer(rmsle)



model = xg.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4,colsample_bytree=0.6,subsample=0.6)

param_test3 = {

 'gamma':[i/10.0for i in range(0,5)]

}

kf = KFold(n_splits=10, shuffle=True, random_state=7) # Define the split - into 2 folds

grid_search = GridSearchCV(model,param_grid = param_test3, scoring=rmsle,n_jobs=-1, cv=kf)

yLabelsLog = np.log1p(yLabels)



grid_result = grid_search.fit(dataTrain, yLabelsLog)

# summarize results



means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

	print("%f (%f) with: %r" % (mean, stdev, param))

print("Best: %f using %s" % ( grid_result.best_score_, grid_result.best_params_))
from sklearn.model_selection import KFold

def rmsle(y, y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))

from sklearn.metrics import fbeta_score, make_scorer

rmsle = make_scorer(rmsle)



model = xg.XGBRegressor(max_depth=9,min_child_weight=1,gamma=0.0,colsample_bytree=0.6,subsample=0.6,

                   learning_rate = 0.01, n_estimators = 400)

param_test3 = {

 'reg_lambda':[i for i in range(0,1)]

}

kf = KFold(n_splits=10, shuffle=True, random_state=7) # Define the split - into 2 folds

grid_search = GridSearchCV(model,param_grid = param_test3, scoring=rmsle,n_jobs=-1, cv=kf)

yLabelsLog = np.log1p(yLabels)



grid_result = grid_search.fit(dataTrain, yLabelsLog)

# summarize results



means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

	print("%f (%f) with: %r" % (mean, stdev, param))

print("Best: %f using %s" % ( grid_result.best_score_, grid_result.best_params_))
# eval_set = [(X_train, y_train), (X_test, y_test)]

# # eval_metric = ['rmsle']

# def rmsle(predictions, dmat):

#     labels = dmat.get_label()

#     diffs = np.log(predictions + 1) - np.log(labels + 1)

#     squared_diffs = np.square(diffs)

#     avg = np.mean(squared_diffs)

#     return ('RMSLE', np.sqrt(avg))

# %time xgr.fit(X_train, y_train, eval_metric= rmsle, eval_set=eval_set, verbose=True)
# results = xgr.evals_result()

# from matplotlib import pyplot

# epochs = len(results['validation_0']['RMSLE'])

# x_axis = range(0, epochs)

# fig, ax = pyplot.subplots()

# ax.plot(x_axis, results['validation_0']['RMSLE'], label='Train')

# ax.plot(x_axis, results['validation_1']['RMSLE'], label='Test')

# ax.legend()

# pyplot.ylabel('LOST FUNCTION : RMSLE')

# pyplot.title('XGBoost REGRESSION Error')

# pyplot.show()
# results = xgr.evals_result()

# from matplotlib import pyplot

# epochs = len(results['validation_0']['rmse'])

# x_axis = range(0, epochs)

# fig, ax = pyplot.subplots()

# ax.plot(x_axis, results['validation_0']['rmse'], label='Train')

# ax.plot(x_axis, results['validation_1']['rmse'], label='Test')

# ax.legend()

# pyplot.ylabel('LOST FUNCTION : RMSE')

# pyplot.title('XGBoost REGRESSION Error')

# pyplot.show()
# eval_set = [ (X_test, y_test)]

# xgr.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=rmsle, eval_set=eval_set, verbose=True)
# # make predictions for test data

# y_pred = xgr.predict(X_test)

# # evaluate predictions

# print ("RMSLE Value For Gradient Boost: ",rmsle(y_test,np.exp(y_pred)))