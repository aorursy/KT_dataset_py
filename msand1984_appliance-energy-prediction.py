# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing, model_selection, metrics

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/KAG_energydata_complete.csv")
data.head()
data.info()
data.describe()
print('The number of rows in dataset is - ' , data.shape[0])

print('The number of columns in dataset is - ' , data.shape[1])
#Number of null values in all columns

data.isnull().sum().sort_values(ascending = True)
from sklearn.model_selection import train_test_split



# 75% of the data is usedfor the training of the models and the rest is used for testing

train, test = train_test_split(data,test_size=0.25,random_state=40)
train.describe()
# Divide the columns based on type for clear column management 



col_temp = ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]



col_hum = ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_6","RH_7","RH_8","RH_9"]



col_weather = ["T_out", "Tdewpoint","RH_out","Press_mm_hg",

                "Windspeed","Visibility"] 

col_light = ["lights"]



col_randoms = ["rv1", "rv2"]



col_target = ["Appliances"]
# Seperate dependent and independent variables 

feature_vars = train[col_temp + col_hum + col_weather + col_light + col_randoms ]

target_vars = train[col_target]
feature_vars.describe()
# Check the distribution of values in lights column

feature_vars.lights.value_counts()
target_vars.describe()
# Due to lot of zero enteries this column is of not much use and will be ignored in rest of the model

_ = feature_vars.drop(['lights'], axis=1 , inplace= True) ;
feature_vars.head(2)
# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# To understand the timeseries variation of the applaince energy consumption

visData = go.Scatter( x= data.date  ,  mode = "lines", y = data.Appliances )

layout = go.Layout(title = 'Appliance energy consumption measurement' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))

fig = go.Figure(data=[visData],layout=layout)



iplot(fig)
# Adding column to mark weekdays (0) and weekends(1) for time series evaluation , 

# decided not to use it for model evaluation as it has least impact



data['WEEKDAY'] = ((pd.to_datetime(data['date']).dt.dayofweek)// 5 == 1).astype(float)

# There are 5472 weekend recordings 

data['WEEKDAY'].value_counts()
# Find rows with weekday 

temp_weekday =  data[data['WEEKDAY'] == 0]

# To understand the timeseries variation of the applaince energy consumption

visData = go.Scatter( x= temp_weekday.date  ,  mode = "lines", y = temp_weekday.Appliances )

layout = go.Layout(title = 'Appliance energy consumption measurement on weekdays' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))

fig = go.Figure(data=[visData],layout=layout)



iplot(fig)
# Find rows with weekday 



temp_weekend =  data[data['WEEKDAY'] == 1]



# To understand the timeseries variation of the applaince energy consumption

visData = go.Scatter( x= temp_weekend.date  ,  mode = "lines", y = temp_weekend.Appliances )

layout = go.Layout(title = 'Appliance energy consumption measurement on weekend' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))

fig = go.Figure(data=[visData],layout=layout)



iplot(fig)
# Histogram of all the features to understand the distribution

feature_vars.hist(bins = 20 , figsize= (12,16)) ;
# focussed displots for RH_6 , RH_out , Visibility , Windspeed due to irregular distribution

f, ax = plt.subplots(2,2,figsize=(12,8))

vis1 = sns.distplot(feature_vars["RH_6"],bins=10, ax= ax[0][0])

vis2 = sns.distplot(feature_vars["RH_out"],bins=10, ax=ax[0][1])

vis3 = sns.distplot(feature_vars["Visibility"],bins=10, ax=ax[1][0])

vis4 = sns.distplot(feature_vars["Windspeed"],bins=10, ax=ax[1][1])
# Distribution of values in Applainces column

f = plt.figure(figsize=(12,5))

plt.xlabel('Appliance consumption in Wh')

plt.ylabel('Frequency')

sns.distplot(target_vars , bins=10 ) ;
#Appliance column range with consumption less than 200 Wh

print('Percentage of the appliance consumption is less than 200 Wh')

print(((target_vars[target_vars <= 200].count()) / (len(target_vars)))*100 )
# Use the weather , temperature , applainces and random column to see the correlation

train_corr = train[col_temp + col_hum + col_weather +col_target+col_randoms]

corr = train_corr.corr()

# Mask the repeated values

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

  

f, ax = plt.subplots(figsize=(16, 14))

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, annot=True, fmt=".2f" , mask=mask,)

    #Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

    #Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

    #show plot

plt.show()
def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



# Function to get top correlations 



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations")

print(get_top_abs_correlations(train_corr, 40))
#Split training dataset into independent and dependent varibales

train_X = train[feature_vars.columns]

train_y = train[target_vars.columns]
#Split testing dataset into independent and dependent varibales

test_X = test[feature_vars.columns]

test_y = test[target_vars.columns]
# Due to conlusion made above below columns are removed

train_X.drop(["rv1","rv2","Visibility","T6","T9"],axis=1 , inplace=True)
# Due to conlusion made above below columns are removed

test_X.drop(["rv1","rv2","Visibility","T6","T9"], axis=1, inplace=True)
train_X.columns
test_X.columns
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()



# Create test and training set by including Appliances column



train = train[list(train_X.columns.values) + col_target ]



test = test[list(test_X.columns.values) + col_target ]



# Create dummy test and training set to hold scaled values



sc_train = pd.DataFrame(columns=train.columns , index=train.index)



sc_train[sc_train.columns] = sc.fit_transform(train)



sc_test= pd.DataFrame(columns=test.columns , index=test.index)



sc_test[sc_test.columns] = sc.fit_transform(test)

sc_train.head()
sc_test.head()
# Remove Appliances column from traininig set



train_X =  sc_train.drop(['Appliances'] , axis=1)

train_y = sc_train['Appliances']



test_X =  sc_test.drop(['Appliances'] , axis=1)

test_y = sc_test['Appliances']
train_X.head()
train_y.head()
from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.neural_network import MLPRegressor

import xgboost as xgb

from sklearn import neighbors

from sklearn.svm import SVR

models = [

           ['Lasso: ', Lasso()],

           ['Ridge: ', Ridge()],

           ['KNeighborsRegressor: ',  neighbors.KNeighborsRegressor()],

           ['SVR:' , SVR(kernel='rbf')],

           ['RandomForest ',RandomForestRegressor()],

           ['ExtraTreeRegressor :',ExtraTreesRegressor()],

           ['GradientBoostingClassifier: ', GradientBoostingRegressor()] ,

           ['XGBRegressor: ', xgb.XGBRegressor()] ,

           ['MLPRegressor: ', MLPRegressor(  activation='relu', solver='adam',learning_rate='adaptive',max_iter=1000,learning_rate_init=0.01,alpha=0.01)]

         ]

# Run all the proposed models and update the information in a list model_data

import time

from math import sqrt

from sklearn.metrics import mean_squared_error



model_data = []

for name,curr_model in models :

    curr_model_data = {}

    curr_model.random_state = 78

    curr_model_data["Name"] = name

    start = time.time()

    curr_model.fit(train_X,train_y)

    end = time.time()

    curr_model_data["Train_Time"] = end - start

    curr_model_data["Train_R2_Score"] = metrics.r2_score(train_y,curr_model.predict(train_X))

    curr_model_data["Test_R2_Score"] = metrics.r2_score(test_y,curr_model.predict(test_X))

    curr_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(test_y,curr_model.predict(test_X)))

    model_data.append(curr_model_data)
model_data
# Convert list to dataframe

df = pd.DataFrame(model_data)
df
df.plot(x="Name", y=['Test_R2_Score' , 'Train_R2_Score' , 'Test_RMSE_Score'], kind="bar" , title = 'R2 Score Results' , figsize= (10,8)) ;
from sklearn.model_selection import GridSearchCV

param_grid = [{

              'max_depth': [80, 150, 200,250],

              'n_estimators' : [100,150,200,250],

              'max_features': ["auto", "sqrt", "log2"]

            }]

reg = ExtraTreesRegressor(random_state=40)

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = reg, param_grid = param_grid, cv = 5, n_jobs = -1 , scoring='r2' , verbose=2)

grid_search.fit(train_X, train_y)
# Tuned parameter set

grid_search.best_params_
# Best possible parameters for ExtraTreesRegressor

grid_search.best_estimator_
# R2 score on training set with tuned parameters



grid_search.best_estimator_.score(train_X,train_y)
# R2 score on test set with tuned parameters

grid_search.best_estimator_.score(test_X,test_y)
# RMSE score on test set with tuned parameters



np.sqrt(mean_squared_error(test_y, grid_search.best_estimator_.predict(test_X)))
# Get sorted list of features in order of importance

feature_indices = np.argsort(grid_search.best_estimator_.feature_importances_)
importances = grid_search.best_estimator_.feature_importances_

indices = np.argsort(importances)[::-1]

names = [train_X.columns[i] for i in indices]

# Create plot

plt.figure(figsize=(10,6))



# Create plot title

plt.title("Feature Importance")



# Add bars

plt.bar(range(train_X.shape[1]), importances[indices])



# Add feature names as x-axis labels

plt.xticks(range(train_X.shape[1]), names, rotation=90)



# Show plot

plt.show()
# Get top 5 most important feature 

names[0:5]
# Get 5 least important feature 

names[-5:]
# Reduce test & training set to 5 feature set

train_important_feature = train_X[names[0:5]]

test_important_feature = test_X[names[0:5]]
# Clone the Gridsearch model with his parameter and fit on reduced dataset



from sklearn.base import clone

cloned_model = clone(grid_search.best_estimator_)

cloned_model.fit(train_important_feature , train_y)
# Reduced dataset scores 



print('Training set R2 Score - ', metrics.r2_score(train_y,cloned_model.predict(train_important_feature)))

print('Testing set R2 Score - ', metrics.r2_score(test_y,cloned_model.predict(test_important_feature)))

print('Testing set RMSE Score - ', np.sqrt(mean_squared_error(test_y, cloned_model.predict(test_important_feature))))
