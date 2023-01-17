# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd,numpy as np

import matplotlib.pyplot as plt    # For plotting

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
df = pd.read_excel('../input/Data_Train.xlsx')

df = df.drop(['Arrival_Time','Additional_Info'],axis = 1)



tf = pd.read_excel('../input/Test_set.xlsx')

tf = tf.drop(['Arrival_Time','Additional_Info'],axis = 1)

df.head()

#tf.head()
df = df[df['Price']<= 20000]

#tf = tf[tf['Price']<= 20000]

df.head()

df.shape


Class = {'IndiGo': 'Economy',

         'GoAir': 'Economy',

         'Vistara': 'Economy',

         'Vistara Premium economy': 'Premium Economy',

         'Air Asia': 'Economy',

         'Trujet': 'Economy',

         'Jet Airways': 'Economy',

         'SpiceJet': 'Economy',

         'Jet Airways Business': 'Business',

         'Air India': 'Economy',

         'Multiple carriers': 'Economy',

         'Multiple carriers Premium economy': 'Premium Economy'}

df['Booking_Class'] = df['Airline'].map(Class)

tf['Booking_Class'] = tf['Airline'].map(Class)
market = {'IndiGo': 41.3,

         'GoAir': 8.4,

         'Vistara': 3.3,

         'Vistara Premium economy': 3.3,

         'Air Asia': 3.3,

         'Trujet': 0.1,

         'Jet Airways': 17.8,

         'SpiceJet': 13.3,

         'Jet Airways Business': 17.8,

         'Air India': 13.5,

         'Multiple carriers': 1,

         'Multiple carriers Premium economy': 1}

df['Market_Share'] = df['Airline'].map(market)

tf['Market_Share'] = tf['Airline'].map(market)


df1 = df.copy() 

df1['Day_of_Booking'] = '1/3/2019'

df1['Day_of_Booking'] = pd.to_datetime(df1['Day_of_Booking'],format='%d/%m/%Y')

df1['Date_of_Journey'] = pd.to_datetime(df1['Date_of_Journey'],format='%d/%m/%Y')

df1['Days_to_Departure'] = (df1['Date_of_Journey'] - df1['Day_of_Booking']).dt.days

df['Days_to_Departure'] = df1['Days_to_Departure']



df2 = tf.copy() 

df2['Day_of_Booking'] = '1/3/2019'

df2['Day_of_Booking'] = pd.to_datetime(df2['Day_of_Booking'],format='%d/%m/%Y')

df2['Date_of_Journey'] = pd.to_datetime(df2['Date_of_Journey'],format='%d/%m/%Y')

df2['Days_to_Departure'] = (df2['Date_of_Journey'] - df2['Day_of_Booking']).dt.days

tf['Days_to_Departure'] = df2['Days_to_Departure']



del df1, df2
df.head()
import pandas_profiling as pp

pp.ProfileReport(df)
df.iloc[:,1] = pd.to_datetime(df.iloc[:,1])

df.iloc[:,6] = df.iloc[:,6].replace("h",'', regex=True)

df.iloc[:,6] = df.iloc[:,6].replace("m",'', regex=True)



tf.iloc[:,1] = pd.to_datetime(tf.iloc[:,1])

tf.iloc[:,6] = tf.iloc[:,6].replace("h",'', regex=True)

tf.iloc[:,6] = tf.iloc[:,6].replace("m",'', regex=True)

# df.iloc[:,6] = pd.to_datetime(df.iloc[:,6],format='%H:%M')

df.dtypes
new = df['Duration'].str.split(" ", n = 2, expand = True) 

df1 = df.copy()

df1[['Duration_hour','Duration_min']] = new

df1[['Duration_hour','Duration_min']] = df1[['Duration_hour','Duration_min']].fillna(0)

df1 = df1.drop(['Duration'],axis = 1)

df1['Dep_Time']= df1['Dep_Time'].str.split(":", n = 1, expand = True)[0]

df1.head()



newt = tf['Duration'].str.split(" ", n = 2, expand = True) 

tf1 = tf.copy()

tf1[['Duration_hour','Duration_min']] = newt

tf1[['Duration_hour','Duration_min']] = tf1[['Duration_hour','Duration_min']].fillna(0)

tf1 = tf1.drop(['Duration'],axis = 1)

tf1['Dep_Time']= tf1['Dep_Time'].str.split(":", n = 1, expand = True)[0]

tf1.head()
df1['DOJ_Day'] = pd.to_datetime(df1.iloc[:,1]).dt.day

df1['DOJ_Month'] = pd.to_datetime(df1.iloc[:,1]).dt.month

df1 = df1.drop(['Date_of_Journey'],axis = 1)

df1.head()



tf1['DOJ_Day'] = pd.to_datetime(tf1.iloc[:,1]).dt.day

tf1['DOJ_Month'] = pd.to_datetime(tf1.iloc[:,1]).dt.month

tf1 = tf1.drop(['Date_of_Journey'],axis = 1)

tf1.head()
df1.iloc[:,[8,9,10,11]] = df1.iloc[:,[10,8,9,11]].astype('int')

df1.iloc[:,[13,11]] = df1.iloc[:,[13,11]].astype('object')

df1.dtypes



tf1.iloc[:,[8,9,7,10]] = tf1.iloc[:,[8,9,7,10]].astype('int')

tf1.iloc[:,[12,11]] = tf1.iloc[:,[12,11]].astype('object')

tf1.dtypes
df1 = pd.get_dummies(df1)

df1.head()



tf1 = pd.get_dummies(tf1)

tf1.head()
X = df1.copy().drop("Price",axis=1) 

y = df["Price"]



## Split the data into trainx, testx, trainy, testy with test_size = 0.20 using sklearn

trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.20)



## Print the shape of X_train, X_test, y_train, y_test

print(trainx.shape)

print(testx.shape)

print(trainy.shape)

print(testy.shape)
testx.head()
from sklearn.preprocessing import StandardScaler



## Scale the numeric attributes

scaler = StandardScaler()

scaler.fit(trainx.iloc[:,:4])



trainx.iloc[:,:4] = scaler.transform(trainx.iloc[:,:4])

testx.iloc[:,:4] = scaler.transform(testx.iloc[:,:4])

tf1.iloc[:,:4] = scaler.transform(tf1.iloc[:,:4])

trainx.head()
from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()



lin_model.fit(trainx,trainy)
lin_train_pred = lin_model.predict(trainx)

lin_test_pred = lin_model.predict(testx)
from sklearn.metrics import mean_absolute_error,mean_squared_error

import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mean_absolute_error(lin_train_pred,trainy))

print(mean_absolute_percentage_error(lin_train_pred,trainy))

print(mean_absolute_error(lin_test_pred,testy))

print(mean_absolute_percentage_error(lin_test_pred,testy))
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha = 0.9)

ridge_model.fit(trainx,trainy)
ridge_train_pred = ridge_model.predict(trainx)

print(mean_absolute_percentage_error(ridge_train_pred,trainy))



ridge_test_pred = ridge_model.predict(testx)

print(mean_absolute_percentage_error(ridge_test_pred,testy))
from yellowbrick.regressor import ResidualsPlot



# Instantiate the linear model and visualizer

ridge = Ridge()

visualizer = ResidualsPlot(ridge)



visualizer.fit(trainx, trainy)  # Fit the training data to the model

visualizer.score(testx , testy)  # Evaluate the model on the test data

visualizer.poof()  # Draw/show/poof the data
visualizer.poof()
from yellowbrick.model_selection import LearningCurve



from sklearn.linear_model import RidgeCV

sizes = np.linspace(0.3, 1.0, 10)



# Create the learning curve visualizer, fit and poof

viz = LearningCurve(RidgeCV(cv = 3), train_sizes=sizes, scoring='r2')

viz.fit(trainx,trainy)

viz.poof()
from sklearn.linear_model import Lasso

Lasso_model = Lasso(alpha = 0.5,max_iter=5000)

%time Lasso_model.fit(trainx,trainy)
Lasso_train_pred = Lasso_model.predict(trainx)

print(mean_absolute_percentage_error(Lasso_train_pred,trainy))



Lasso_test_pred = Lasso_model.predict(testx)

print(mean_absolute_percentage_error(Lasso_test_pred,testy))
from sklearn.neighbors import KNeighborsRegressor

KNN = KNeighborsRegressor(n_neighbors=6,metric='euclidean')

%time KNN.fit(trainx,trainy) 
import math

%time KNN_train_pred = KNN.predict(trainx)

print(mean_absolute_percentage_error(KNN_train_pred,trainy))



%time KNN_test_pred = KNN.predict(testx)

print(mean_absolute_percentage_error(KNN_test_pred,testy))



print(math.sqrt(mean_squared_error(KNN_test_pred,testy)))
from sklearn.svm import SVR

SVR = SVR(gamma='scale', C=2.0, epsilon=0.1,kernel='poly')

%time SVR.fit(trainx,trainy)
SVR_train_pred = SVR.predict(trainx)

print(mean_absolute_percentage_error(SVR_train_pred,trainy))



SVR_test_pred = SVR.predict(testx)

print(mean_absolute_percentage_error(SVR_test_pred,testy))



print(math.sqrt(mean_squared_error(SVR_test_pred,testy)))
from sklearn.ensemble import RandomForestRegressor



regr = RandomForestRegressor(n_jobs=-1,min_samples_leaf=3, min_samples_split=2,max_depth=51,n_estimators=400)

%time regr.fit(trainx, trainy)  
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



rfr_train_pred = regr.predict(trainx)

print(mean_squared_error(trainy, rfr_train_pred))

from math import sqrt

print(sqrt(mean_squared_error(trainy, rfr_train_pred)))

print(mean_absolute_error(trainy, rfr_train_pred))



rfr_test_pred = regr.predict(testx)

print(mean_squared_error(testy, rfr_test_pred))

from math import sqrt

print(sqrt(mean_squared_error(testy, rfr_test_pred)))

print(mean_absolute_error(testy, rfr_test_pred))



import numpy as np



def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



print(mean_absolute_percentage_error(trainy,rfr_train_pred))

print(mean_absolute_percentage_error(testy,rfr_test_pred))



def rmsle(y_pred, y_test) : 

    assert len(y_test) == len(y_pred)

    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))



print(rmsle(rfr_train_pred,trainy))

print(rmsle(rfr_test_pred,testy))
from yellowbrick.model_selection import ValidationCurve



RFviz = ValidationCurve(

    RandomForestRegressor(), param_name="max_depth",

    param_range=np.arange(2, 20), cv=3, scoring='neg_mean_absolute_error'

)



# Fit and poof the visualizer

RFviz.fit(trainx, trainy)

RFviz.poof()
from sklearn.model_selection import RandomizedSearchCV





rfr_grid2 = RandomForestRegressor(n_jobs=-1, max_features='auto',oob_score=True)

 

# Set parameters for the grid search



 

param_grid2 = {"n_estimators" : [250,500],

           "max_depth" : [12,15],

           "min_samples_leaf" : [1,3],

           'min_samples_split' : [2,3],

           'max_features' : ['auto','sqrt']}

 

rfr_cv_grid2 = RandomizedSearchCV(estimator = rfr_grid2, param_distributions = param_grid2, cv = 5, n_iter=10)

%time rfr_cv_grid2.fit(trainx, trainy)

rfr_cv_grid2.best_estimator_
rfr_cv_grid2.best_score_
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



rfr2_train_pred = rfr_cv_grid2.predict(trainx)

print(mean_squared_error(trainy, rfr2_train_pred))

from math import sqrt

print(sqrt(mean_squared_error(trainy, rfr2_train_pred)))

print(mean_absolute_error(trainy, rfr2_train_pred))



rfr2_test_pred = rfr_cv_grid2.predict(testx)

print(mean_squared_error(testy, rfr2_test_pred))

from math import sqrt

print(sqrt(mean_squared_error(testy, rfr2_test_pred)))

print(mean_absolute_error(testy, rfr2_test_pred))



print(rmsle(rfr2_train_pred,trainy))

print(rmsle(rfr2_test_pred,testy))



print(mean_absolute_percentage_error(trainy,rfr2_train_pred))

print(mean_absolute_percentage_error(testy,rfr2_test_pred))
from sklearn.ensemble import AdaBoostRegressor

ABR = AdaBoostRegressor(base_estimator=None, learning_rate=0.1, loss='exponential',

        n_estimators= 100, random_state=1)

%time ABR.fit(trainx,trainy)
ABR_train_pred = ABR.predict(trainx)

print(mean_absolute_percentage_error(ABR_train_pred,trainy))



ABR_test_pred = ABR.predict(testx)

print(mean_absolute_percentage_error(ABR_test_pred,testy))



print(math.sqrt(mean_squared_error(ABR_test_pred,testy)))
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=6, n_estimators=2000,learning_rate=0.05)

# gbrt.fit(trainx, trainy)

# errors = [mean_squared_error(testy, y_pred)

#  for y_pred in gbrt.staged_predict(testx)]

# bst_n_estimators = np.argmin(errors)

# gbrt_best = GradientBoostingRegressor(max_depth=12,n_estimators=bst_n_estimators)

%time gbrt.fit(trainx, trainy)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



gbrt_train_pred = gbrt.predict(trainx)

print(mean_squared_error(trainy, gbrt_train_pred))

from math import sqrt

print(sqrt(mean_squared_error(trainy, gbrt_train_pred)))

print(mean_absolute_error(trainy, gbrt_train_pred))



gbrt_test_pred = gbrt.predict(testx)

print(mean_squared_error(testy, gbrt_test_pred))

from math import sqrt

print(sqrt(mean_squared_error(testy, gbrt_test_pred)))

print(mean_absolute_error(testy, gbrt_test_pred))



print(rmsle(gbrt_train_pred,trainy))

print(rmsle(gbrt_test_pred,testy))



print(mean_absolute_percentage_error(trainy,gbrt_train_pred))

print(mean_absolute_percentage_error(testy,gbrt_test_pred))
import xgboost as xgb

from sklearn.metrics import mean_squared_error



xg_reg = xgb.XGBRegressor(max_depth=8, 

                   learning_rate=0.15, 

                   n_estimators=210, 

                   silent=False, 

                   objective='reg:linear', 

                   booster='gbtree', 

                   n_jobs=1, 

                   nthread=None, 

                   gamma=0, 

                   min_child_weight=0, 

                   max_delta_step=0, 

                   subsample=1, 

                   colsample_bytree=1, 

                   colsample_bylevel=1, 

                   reg_alpha=0.1, 

                   reg_lambda=0.1, 

                   scale_pos_weight=1, 

                   base_score=0.5, 

                   random_state=0, )

%time xg_reg.fit(trainx,trainy)
xg_reg_train_pred = xg_reg.predict(trainx)

print(mean_squared_error(trainy, xg_reg_train_pred))

from math import sqrt

print(sqrt(mean_squared_error(trainy, xg_reg_train_pred)))

#print(mean_absolute_error(trainy, xg_reg_train_pred))



xg_reg_test_pred = xg_reg.predict(testx)

print(mean_squared_error(testy, xg_reg_test_pred))

from math import sqrt

print(sqrt(mean_squared_error(testy, xg_reg_test_pred)))

#print(mean_absolute_error(testy, xg_reg_test_pred))



print(rmsle(xg_reg_train_pred,trainy))

print(rmsle(xg_reg_test_pred,testy))



print(mean_absolute_percentage_error(trainy,xg_reg_train_pred))

print(mean_absolute_percentage_error(testy,xg_reg_test_pred))
import xgboost as xgb

from sklearn.model_selection import validation_curve

default_params = {

    'objective': 'reg:linear',

    'max_depth': 8,

    'learning_rate': 0.1

    }



n_estimators_range = np.linspace(1, 240, 30).astype('int')



train_scores, test_scores = validation_curve(

    xgb.XGBRegressor(**default_params),

    trainx, trainy,

    param_name = 'n_estimators',

    param_range = n_estimators_range,

    cv=2,

    scoring='neg_mean_squared_log_error'

)


%matplotlib inline

train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)



fig = plt.figure(figsize=(10, 6), dpi=100)



plt.title("Validation Curve with XGBoost (eta = 0.3)")

plt.xlabel("number of trees")

plt.ylabel("neg_mean_absolute_error")

plt.ylim(-0.4,0.25)



plt.plot(n_estimators_range,

             train_scores_mean,

             label="Training score",

             color="r")



plt.plot(n_estimators_range,

             test_scores_mean, 

             label="Cross-validation score",

             color="g")



plt.fill_between(n_estimators_range, 

                 train_scores_mean - train_scores_std,

                 train_scores_mean + train_scores_std, 

                 alpha=0.2, color="r")



plt.fill_between(n_estimators_range,

                 test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std,

                 alpha=0.2, color="g")



plt.axhline(y=1, color='k', ls='dashed')



plt.legend(loc="best")

plt.show()



i = np.argmax(test_scores_mean)

print("Best cross-validation result ({0:.2f}) obtained for {1} trees".format(test_scores_mean[i], n_estimators_range[i]))
import lightgbm as lgb

train_data=lgb.Dataset(trainx,label=trainy)

params = {'objective': 'regression',

         'boosting': 'gbdt',

         'num_iterations': 8000,   

         'learning_rate': 0.01,  

         'num_leaves': 40,  

         'max_depth': 24,   

         'min_data_in_leaf':6,  

         'max_bin': 4, 

         'metric': 'mape'

         }

%time lgbmodel= lgb.train(params, train_data)
lgb_reg_train_pred = lgbmodel.predict(trainx)

print(mean_squared_error(trainy, lgb_reg_train_pred))

from math import sqrt

print(sqrt(mean_squared_error(trainy, lgb_reg_train_pred)))

#print(mean_absolute_error(trainy, lgb_reg_train_pred))



lgb_reg_test_pred = lgbmodel.predict(testx)

print(mean_squared_error(testy, lgb_reg_test_pred))

from math import sqrt

print(sqrt(mean_squared_error(testy, lgb_reg_test_pred)))

#print(mean_absolute_error(testy, lgb_reg_test_pred))



print(rmsle(lgb_reg_train_pred,trainy))

print(rmsle(lgb_reg_test_pred,testy))



print(mean_absolute_percentage_error(trainy,lgb_reg_train_pred))

print(mean_absolute_percentage_error(testy,lgb_reg_test_pred))



#print('RMSLE:', sqrt(mean_squared_log_error(np.exp(testy), np.exp(lgb_reg_test_pred))))
sub_pred = 0.4*(xg_reg_test_pred)+0.6*(rfr_test_pred)
print(mean_absolute_percentage_error(testy,sub_pred))