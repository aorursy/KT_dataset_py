# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

# pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from pathlib import Path

data_dir = Path('../input/covid19-global-forecasting-week-1')

pio.templates.default = 'ggplot2'
# from sklearn.compose import ColumnTransformer

# from sklearn.pipeline import Pipeline

# from sklearn.impute import SimpleImputer

# from sklearn.preprocessing import OneHotEncoder



# # Preprocessing for numerical data

# numerical_transformer = SimpleImputer(strategy='constant')



# # Preprocessing for categorical data

# categorical_transformer = Pipeline(steps=[

#     ('imputer', SimpleImputer(strategy='most_frequent')),

#     ('onehot', OneHotEncoder(handle_unknown='ignore'))

# ])



# # Bundle preprocessing for numerical and categorical data

# preprocessor = ColumnTransformer(

#     transformers=[

#         ('num', numerical_transformer, numerical_cols),

#         ('cat', categorical_transformer, categorical_cols)

#     ])
from sklearn.model_selection import train_test_split



# Read the data

data = pd.read_csv(data_dir/'train.csv') #, index_col='Id') 

data_test = pd.read_csv(data_dir/'test.csv') #, index_col='Id')



day_min = pd.to_datetime( data['Date'].min(),format='%Y-%m-%d')

t = (pd.DatetimeIndex(data['Date']) - day_min).days

pd.DatetimeIndex(data['Date']).dayofweek
l_use_date_int = True

l_use_days_num = True



methods = ['use_date_int','use_days_num','use_converter']

c_method = methods[1]



if c_method == methods[0]:



    data["Date"] = data["Date"].apply(lambda x: x.replace("-",""))

    data["Date"]  = data["Date"].astype(int)



    data_test["Date"] = data_test["Date"].apply(lambda x: x.replace("-",""))

    data_test["Date"]  = data_test["Date"].astype(int)



    features = ['Lat','Long','Date']



elif c_method == methods[1]:

    day_min = pd.to_datetime( data['Date'].min(),format='%Y-%m-%d')

    t = (pd.DatetimeIndex(data['Date']) - day_min).days

    

    data.loc[:,'days'] = pd.Series(t)



    t = (pd.DatetimeIndex(data_test['Date']) - day_min).days

    

    data_test.loc[:,'days'] = pd.Series(t)



    features = ['Lat','Long','days']

elif c_method == methods[2]:

    # set the time observation 

    t = (pd.DatetimeIndex(data['Date']) - pd.DatetimeIndex([data_test['Date'].min()])[0]).days



    # data_test = data[data['Date']<data_test['Date'].min()]

    # data_valid = data[data['Date']>=data_test['Date'].min()]



    data['year'] = pd.DatetimeIndex(data['Date']).year

    data['month'] = pd.DatetimeIndex(data['Date']).month

    data['day'] = pd.DatetimeIndex(data['Date']).day

    data_test['year'] = pd.DatetimeIndex(data_test['Date']).year

    data_test['month'] = pd.DatetimeIndex(data_test['Date']).month

    data_test['day'] = pd.DatetimeIndex(data_test['Date']).day



    features = ['Lat','Long','year','month','day']



# Add day of week



features += ['dayofweek']



data['dayofweek'] = pd.DatetimeIndex(data['Date']).dayofweek

data_test['dayofweek'] = pd.DatetimeIndex(data_test['Date']).dayofweek





X = data[features]

X1 = data[features]

y = data.ConfirmedCases

y1 = data.Fatalities



X_test = data_test[features]



# Break off validation set from training data

# X_train = X[t<0]

# X_valid = X[t>=0]

# y_train = y[t<0]

# y_valid = y[t>=0]



X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)

# Break off validation set from training data

X1_train, X1_valid, y1_train, y1_valid = train_test_split(X1, y1,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in data.columns if

                    data[cname].nunique() < 10 and 

                    data[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X.columns if 

                X[cname].dtype in ['int64', 'float64']]



print(categorical_cols, numerical_cols)
[cname for cname in data.columns] # if

#                     data[cname].nunique() < 10 and 

#                     data[cname].dtype == "object"]

# data['Country/Region'].nunique()

for cname in data.columns:

    print(cname, data[cname].nunique() < 10, data[cname].dtype == "object" )
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor  



model0 = RandomForestRegressor(n_estimators=110, random_state=0)

model1 = RandomForestRegressor(n_estimators=50, random_state=0)



model0 = DecisionTreeClassifier(criterion='entropy')

model1 = DecisionTreeClassifier(criterion='entropy')



model0 = DecisionTreeRegressor(random_state = 0) 

model1 = DecisionTreeRegressor(random_state = 0) 





my_pipeline0 = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

    ('model', model0)

])



my_pipeline1 = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

    ('model', model1)

])
from sklearn.model_selection import cross_val_score



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(my_pipeline0, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("Average MAE score:", scores.mean())
# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(my_pipeline1, X, y1,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("Average MAE score:", scores.mean())
def get_score(yy, n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    # Replace this body with your own code

    my_pipeline = Pipeline(steps=[

                    ('preprocessor', SimpleImputer()),

                    ('model',RandomForestRegressor(n_estimators=n_estimators,random_state=0))

                    ])

    score_cross_valids = -1 * cross_val_score(my_pipeline, X, yy,

                              cv=3,

                              scoring='neg_mean_absolute_error')

    return score_cross_valids.mean()

# results0 = {i: get_score(y,i) for i in range(50,450,50)}

# results1 = {i: get_score(y1,i) for i in range(50,450,50)}

# plt.plot(results0.keys(),results0.values() )

# plt.plot(results1.keys(),results1.values() )

# for i in range(50,170,20):

#     print(i, get_score(y,i))
# for i in range(1,16):

#     print(i, get_score(y1,i))
# results1
my_pipeline0.fit(X, y)

my_pipeline1.fit(X, y1)

rf_predictions = my_pipeline0.predict(X_valid)

rf_val_mae = mean_absolute_error(rf_predictions, y_valid)



print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

# preds = my_pipeline.predict(X_test)

test_preds0 = my_pipeline0.predict(X_test)

test_preds1 = my_pipeline1.predict(X_test)



t0 = np.round(test_preds0).astype(int)

t1 = np.round(test_preds1).astype(int)



# test_preds0
output = pd.DataFrame({'ForecastId': data_test.ForecastId,

                       'ConfirmedCases':t0,

                       'Fatalities': t1})

output.to_csv('submission.csv', index=False)
data['log_ConfirmedCases'] = np.log(data['ConfirmedCases']+1)

data['log_Fatalities'] = np.log(data['Fatalities']+1)

y_pred2 = data['log_ConfirmedCases']

y1_pred2 = data['log_Fatalities']







# Break off validation set from training data



day_min = pd.to_datetime( data['Date'].min(),format='%Y-%m-%d')

t = data['days'] < data_test['days'].min()



X_train = X[t]

X_valid = X[~t]

y_train = y_pred2[t]

y_valid = y_pred2[~t]



# def RMSE(y_pred,y_valid):

#     return np
model0 = RandomForestRegressor(n_estimators=300, random_state=0,verbose=True)

model1 = RandomForestRegressor(n_estimators=100, random_state=0)



my_pipeline0 = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

    ('model', model0)

])



my_pipeline1 = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

    ('model', model1)

])

my_pipeline0.fit(X_train, y_train) #,verbose=True)
rf_predictions = my_pipeline0.predict(X_valid)

rf_val_mae = mean_absolute_error(rf_predictions, y_valid)



print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
# t0 = data['Country/Region']=='Spain'

# lat,long = data[t0].iloc[0,:][['Lat','Long']].values



# t = (X_train['Lat']== lat) & (X_train['Long']== long)

# plt.plot(X_train[t]['days'],np.exp(y_train[t])+1)

# t = (X_valid['Lat']== lat) & (X_valid['Long']== long)

# plt.plot(X_valid[t]['days'],np.exp(y_valid[t])+1)

# plt.plot(X_valid[t]['days'],np.exp(rf_predictions[t])+1)

# # t = (X['Lat']== lat) & (X['Long']== long)

# # plt.plot(X[t]['days'],test_preds0[t])

# rf_predictions
def get_score1(n_est):

    model = RandomForestRegressor(n_estimators=n_est, random_state=0)



    my_pipeline = Pipeline(steps=[

                            ('preprocessor', SimpleImputer()),

                            ('model', model)

                            ])

    score_cross_valids = -1 * cross_val_score(my_pipeline0, X_train, y_train,

                              cv=5,

                              scoring='neg_mean_absolute_error')

    return score_cross_valids.mean()
# res = [get_score1(i) for i in [50,100,500,5000]]
my_pipeline0.fit(X, y_pred2) #,verbose=True)

my_pipeline1.fit(X, y1_pred2) #,verbose=True)

# preds = my_pipeline.predict(X_test)

test_preds2 = my_pipeline0.predict(X_test)

test_preds3 = my_pipeline1.predict(X_test)



t0 = np.round(np.exp(test_preds2)-1).astype(int)

t1 = np.round(np.exp(test_preds3)-1).astype(int)

# plt.hist(np.log(test_preds0+1),bins=30)

# plt.hist(test_preds2,bins=30,alpha=0.7)
output = pd.DataFrame({'ForecastId': data_test.ForecastId,

                       'ConfirmedCases':t0,

                       'Fatalities': t1})

output.to_csv('submission.csv', index=False)