#Import Libraries

import sys 

import pandas as pd 

import matplotlib 

import numpy as np 

import scipy as sc

import IPython

from IPython import display 

import sklearn



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)



#Algorithms

from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb

import lightgbm as lgb





from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns





%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8





#Import Data

data_train = pd.read_csv('../input/the-consulting-club-analytics-challenge/Train.csv')

data_test  = pd.read_csv('../input/the-consulting-club-analytics-challenge/Test.csv')



data_submit = pd.read_csv('../input/the-consulting-club-analytics-challenge/SampleSubmission.csv')



data_train.head()

data_test.head()



#preview data

print (data_train.info())

print (data_test.info()) 
#Plot Graphs and analyze data

#Function to plot Graph of each attribute vs target value

def PlotAttributeVsTarget(df,target):

    for y in df.columns:

        if y!= target:

            plt.figure(figsize=(12,5))

            sns.scatterplot(x=data_train[y],y=data_train[target])

            PlotAttributeVsTarget(data_train,'price_usd')
#Function to plot Graph of each attribute(Categorical) vs target value

def PlotAttributeCategoricalVsTarget(df,target):

    for y in df.columns:

        if (y!= target and df[y].dtypes != 'int64' and df[y].dtypes != 'float64'):

            plt.figure(figsize=(12,5))

            sns.boxplot(x=data_train[y],y=data_train[target])

            PlotAttributeCategoricalVsTarget(data_train,'price_usd')
#Outliers

#After analysing all the graph 

#carefully we find some outliers in odometer_value, duration_listed

#We gonna delete these values

print('Shape Of old df:',data_train.shape)



plt.figure(figsize=(6,4))

sns.scatterplot(x=data_train['odometer_value'],y=data_train['price_usd'])

data_train.drop(data_train[data_train['odometer_value'] > 650000].index, inplace = True)

plt.figure(figsize=(6,4))

sns.scatterplot(x=data_train['odometer_value'],y=data_train['price_usd'])

print('Shape Of new df:',data_train.shape)





plt.figure(figsize=(6,4))

sns.scatterplot(x=data_train['duration_listed'],y=data_train['price_usd'])

data_train.drop(data_train[(data_train['duration_listed'] > 1000) & (data_train['price_usd'] < 40000)].index, inplace = True)

plt.figure(figsize=(6,4))

sns.scatterplot(x=data_train['duration_listed'],y=data_train['price_usd'])

print('Shape Of new df:',data_train.shape)

#Add dfs

data_train_test = pd.concat([data_train, data_test]) #We have to split the data later (37860 in train )



#Drop col with same value above certain percentage level

def getuniquevalues(df,percent):

    totalrow = df.shape[0]

    dic = {}

    for col in df.columns:

        count = df[col].value_counts().max()

        per = (count*100)/totalrow

        if per > percent:

            dic[col] = per

    return dic



Columns_Dic = getuniquevalues(data_train_test,95)

print('Number of col with same values: ',len(Columns_Dic))



#Col with values sorted

import operator

sorted_x = sorted(Columns_Dic.items(), key=operator.itemgetter(1))

sorted_x.reverse()

sorted_x
#Deleting columns

Columns_List= list(Columns_Dic.keys())

print('Shape Of old df:',data_train_test.shape)

data_train_test = data_train_test.drop(Columns_List,axis=1)

print('Shape Of new df:',data_train_test.shape)
#Dealing with null values

nulls = np.sum(data_train_test.isnull())

nullcols = nulls.loc[(nulls != 0)]

dtypes = data_train_test.dtypes

dtypes2 = dtypes.loc[(nulls != 0)]

info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)

print(info)

print("There are", len(nullcols), "columns with missing values")

print('Null in  data:',data_train_test.isnull().sum().sum())

print('Shape of df:', data_train_test.shape)
#We have to fill missing values of price_usd and engine_capacity

#Filling missing values



#price_usd

data_train_test['price_usd'].fillna(data_train_test['price_usd'].median(), inplace = True)

#engine_capacity

data_train_test['engine_capacity'].fillna(data_train_test['engine_capacity'].median(), inplace = True)
#price_usd

sns.distplot(data_train['price_usd'])

data_train['price_usd'].mean()

data_train['price_usd'].median()
#To determine skewness in the dataset



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in data_train_test.columns:

    if data_train_test[i].dtype in numeric_dtypes: 

        numerics2.append(i)



skew_features = data_train_test[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

skews = pd.DataFrame({'skew':skew_features})

skews
high_skew = skew_features[skew_features > 0.5]

high_skew = high_skew

skew_index = high_skew.index



for i in skew_index:

    data_train_test[i]= boxcox1p(data_train_test[i], boxcox_normmax(data_train_test[i]+1))



        

skew_features2 = data_train_test[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

skews2 = pd.DataFrame({'skew':skew_features2})

skews2
#Get dummies

print(data_train_test.shape)

data_train_test = pd.get_dummies(data_train_test,drop_first = True).reset_index(drop=True)

print(data_train_test.shape)
#Split Data        

df_train = data_train_test.iloc[:1458,:] #We have 37860 rows in train dataframe

df_test = data_train_test.iloc[1458:,:]          

print('Shape of Train df:', df_train.shape,'\nShape of Test df:', df_test.shape, '\nShape of train_test df:', data_train_test.shape)
#Assign X and Y

X_train = df_train.iloc[:,df_train.columns != 'price_usd']

y_train = df_train.iloc[:,df_test.columns == 'price_usd']

X_test = df_test.iloc[:,df_test.columns != 'price_usd']
#Gradient Boosting Regression 

model_GB = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

#XGBoost 

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

#LightGBM :

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf =11)
"""model_GB.fit(X_train,y_train.values.ravel())

model_xgb.fit(X_train,y_train.values.ravel())

model_lgb.fit(X_train,y_train.values.ravel())"""
"""y_pred_model_GB = model_GB.predict(X_test)

y_pred_model_xgb = model_xgb.predict(X_test)

y_pred_model_lgb = model_lgb.predict(X_test)



y_pred=(y_pred_model_GB+y_pred_model_xgb+y_pred_model_lgb)/3



data_submit['price_usd'] = y_pred

data_submit.to_csv("submission.csv", index=False)"""