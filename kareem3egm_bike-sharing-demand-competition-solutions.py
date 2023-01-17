# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Import Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt 

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error 

from sklearn.metrics import mean_absolute_error 

#----------------------------------------------------



#----------------------------------------------------

train_st = pd.read_csv('../input/bike-sharing-demand/train.csv')

test_st = pd.read_csv('../input/bike-sharing-demand/test.csv')

train_st.head()
print(train_st.columns.values)
print("train shape:",train_st.shape)

print("test shape :",test_st.shape)
train_st.info()

print('_______________________________________________')

test_st.info()
train_st.describe()
from pandas_profiling import ProfileReport 



profile = ProfileReport( train_st, title='Pandas profiling report ' , html={'style':{'full_width':True}})



profile.to_notebook_iframe()
sns.pairplot(train_st ,kind="reg")
season=pd.get_dummies(train_st['season'],prefix='season')

dtrain_stf=pd.concat([train_st,season],axis=1)

train_st.head()

season=pd.get_dummies(test_st['season'],prefix='season')

test_st=pd.concat([test_st,season],axis=1)

test_st.head()
weather=pd.get_dummies(train_st['weather'],prefix='weather')

train_st=pd.concat([train_st,weather],axis=1)

train_st.head()

weather=pd.get_dummies(test_st['weather'],prefix='weather')

test_st=pd.concat([test_st,weather],axis=1)

test_st.head()
train_st.drop(['season','weather'],inplace=True,axis=1)

train_st.head()

test_st.drop(['season','weather'],inplace=True,axis=1)

test_st.head()

train_st["hour"] = [t.hour for t in pd.DatetimeIndex(train_st.datetime)]

train_st["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_st.datetime)]

train_st["month"] = [t.month for t in pd.DatetimeIndex(train_st.datetime)]

train_st['year'] = [t.year for t in pd.DatetimeIndex(train_st.datetime)]

train_st['year'] = train_st['year'].map({2011:0, 2012:1})

train_st.head()
test_st["hour"] = [t.hour for t in pd.DatetimeIndex(test_st.datetime)]

test_st["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_st.datetime)]

test_st["month"] = [t.month for t in pd.DatetimeIndex(test_st.datetime)]

test_st['year'] = [t.year for t in pd.DatetimeIndex(test_st.datetime)]

test_st['year'] = test_st['year'].map({2011:0, 2012:1})

test_st.head()
# now can drop datetime column.

train_st.drop('datetime',axis=1,inplace=True)

train_st.head()
print(train_st.columns.values)
train_st.shape
x_train,x_test,y_train,y_test=train_test_split(train_st.drop('count',axis=1),train_st['count'],test_size=0.25,random_state=42)


#----------------------------------------------------

#Applying Random Forest Regressor Model 



'''

sklearn.ensemble.RandomForestRegressor(n_estimators='warn', criterion=’mse’, max_depth=None,

                                       min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0.0,

                                       max_features=’auto’, max_leaf_nodes=None,min_impurity_decrease=0.0,

                                       min_impurity_split=None, bootstrap=True,oob_score=False, n_jobs=None,

                                       random_state=None, verbose=0,warm_start=False)

'''



RandomForestRegressorModel = RandomForestRegressor(n_estimators=100,max_depth=5, random_state=33)

RandomForestRegressorModel.fit(x_train, y_train)



#Calculating Details

print('Random Forest Regressor Train Score is : ' , RandomForestRegressorModel.score(x_train, y_train))

print('Random Forest Regressor Test Score is : ' , RandomForestRegressorModel.score(x_test, y_test))

print('Random Forest Regressor No. of features are : ' , RandomForestRegressorModel.n_features_)

print('----------------------------------------------------')



#Calculating Prediction

y_pred = RandomForestRegressorModel.predict(x_test)

print('Predicted Value for Random Forest Regressor is : ' , y_pred[:10])




#----------------------------------------------------

#Calculating Mean Squared Error

MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values

print('Mean Squared Error Value is : ', MSEValue)



#----------------------------------------------------

#Calculating Mean Absolute Error

MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values

print('Mean Absolute Error Value is : ', MAEValue)
pred=RandomForestRegressorModel.predict(test_st.drop('datetime',axis=1))

d={'datetime':test['datetime'],'count':pred}

ans=pd.DataFrame(d)

ans.to_csv('sampleSubmission.csv',index=False)