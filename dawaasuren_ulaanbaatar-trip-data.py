# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import Ridge

from sklearn import linear_model

from sklearn import svm

from math import sqrt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_excel('../input/UB_data.xlsx')

weather_data = pd.read_excel('../input/UB_weather.xlsx')

train_data = pd.merge(train_data,weather_data,on="Date",how="left")

data = pd.DataFrame(train_data)

data = data.dropna(axis=0, how='any')

print(data.columns)
def calculateCorelation(data):

    corr = data.corr(method = 'pearson')

    sns.heatmap(corr, annot=True, fmt = ".2f")
def selector1(data):

    input_data = pd.DataFrame(data, columns =  ['Total_distance','Start_hour','Average_speed','Temperature','Humidity','Pressure','Wind','Visibility','Precipitation','Events'])

    return input_data
def selector2(data):

    input_data = pd.DataFrame(data, columns =  ['Origin_lng','Origin_lat','Destination_lng','Destination_lat','Total_distance','Average_speed','Temperature','Humidity','Pressure','Wind','Visibility','Precipitation','Events'])

    return input_data
def selector3(data):

    input_data = pd.DataFrame(data, columns =  ['Origin_lng','Origin_lat','Destination_lng','Destination_lat','Average_speed','Start_hour','Temperature','Humidity','Pressure','Wind','Visibility','Precipitation','Events'])

    return input_data
def selector4(data):

    input_data = pd.DataFrame(data, columns =  ['Origin_lng','Origin_lat','Destination_lng','Destination_lat','Total_distance','Start_hour','Temperature','Humidity','Pressure','Wind','Visibility','Precipitation','Events'])

    return input_data
def selector5(data):

    input_data = pd.DataFrame(data, columns =  ['Origin_lng','Origin_lat','Destination_lng','Destination_lat','Total_distance','Average_speed','Start_hour'])

    return input_data
def selector6(data):

    input_data = pd.DataFrame(data, columns =  ['Origin_lng','Origin_lat','Destination_lng','Destination_lat','Total_distance','Average_speed','Start_hour','Temperature','Humidity','Pressure','Wind','Visibility','Precipitation','Events'])

    return input_data
def training(train_data,label,alg):

    X = np.array(train_data)

    y = np.array(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    alg.fit(X_train, y_train)

    prediction = alg.predict(X_test)

    r2 = r2_score(y_test, prediction)

    MSE = mean_squared_error(y_test,prediction)

    RMSE = sqrt(MSE/60)

    print('R2 score:', r2)

    print('MSE:', MSE)

    print('RMSE:',RMSE)
calculateCorelation(data)
calculateCorelation(pd.DataFrame(data, columns=['Temperature','Humidity','Pressure','Wind','Visibility','Precipitation','Events','Total_time']))
calculateCorelation(pd.DataFrame(data, columns=['Humidity','Pressure','Events','Total_time']))
calculateCorelation(pd.DataFrame(data, columns=['Temperature','Wind','Visibility','Precipitation','Total_time']))
calculateCorelation(pd.DataFrame(data, columns=['Total_distance','Average_speed','Total_time']))
linear_regressor = linear_model.LinearRegression()

training(selector1(data),data['Total_time'],linear_regressor)
training(selector2(data),data['Total_time'],linear_regressor)
training(selector3(data),data['Total_time'],linear_regressor)
training(selector4(data),data['Total_time'],linear_regressor)
training(selector5(data),data['Total_time'],linear_regressor)
training(selector6(data),data['Total_time'],linear_regressor)
random_forest = RandomForestRegressor(max_depth = 2, random_state=0)

training(selector1(data),data['Total_time'],random_forest)

training(selector2(data),data['Total_time'],random_forest)
training(selector3(data),data['Total_time'],random_forest)
training(selector4(data),data['Total_time'],random_forest)
training(selector5(data),data['Total_time'],random_forest)
training(selector6(data),data['Total_time'],random_forest)
gradient_boosting = GradientBoostingRegressor(alpha=0.6,criterion='friedman_mse',

                                             init=None, learning_rate = 0.1, loss='ls', max_depth = 3,

                                             max_features=None, max_leaf_nodes = None, min_samples_leaf=1,

                                             min_samples_split=2, min_weight_fraction_leaf=0.0,

                                             n_estimators=100, presort='auto', random_state=None,

                                             subsample=1.0, verbose=0, warm_start=False)

training(selector1(data),data['Total_time'],gradient_boosting)
training(selector2(data),data['Total_time'],gradient_boosting)
training(selector3(data),data['Total_time'],gradient_boosting)
training(selector4(data),data['Total_time'],gradient_boosting)
training(selector5(data),data['Total_time'],gradient_boosting)
training(selector6(data),data['Total_time'],gradient_boosting)
ridge = Ridge(alpha=0.98, copy_X=True, fit_intercept=True,

             max_iter=None, normalize=False, random_state=None,solver='auto',

             tol=0.001)

training(selector1(data),data['Total_time'],ridge)
training(selector2(data),data['Total_time'],ridge)
training(selector3(data),data['Total_time'],ridge)
training(selector4(data),data['Total_time'],ridge)
training(selector5(data),data['Total_time'],ridge)
training(selector6(data),data['Total_time'],ridge)