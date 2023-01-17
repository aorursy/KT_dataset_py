# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import matplotlib.pyplot as plt
import datetime as dt
import numpy, scipy, matplotlib
from scipy.optimize import curve_fit
import warnings
from math import exp, e

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Prepare Trian Data
train_path = "/kaggle/input/covid19-global-forecasting-week-3/train.csv"
train_data = pd.read_csv(train_path, parse_dates=['Date'])
#train_data = pd.read_csv(train_path, index_col="Date", parse_dates=True)

# Check the Number of missing values in each column of training data
missing_val_count_by_column = (train_data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
train_data.head()
#Prepare Test Data
test_path = "/kaggle/input/covid19-global-forecasting-week-3/test.csv"
test_data = pd.read_csv(test_path)
test_data1 = pd.read_csv(test_path, index_col="Date", parse_dates=True)
#Logistic Regression Mudule
def train_logistic(i):
    #Prepare Train Data
    train_data_1 = train_data.loc[train_data["Country_Region"].isin([i])]
    train_data_2 = train_data_1.sort_values(by="Date")
    train_data_3 = train_data_2.set_index('Date')
    Date=[]
    ConfirmedCases=[]
    Fatalities=[]
    X_1=[]
    loop=1
    Date = train_data_3.index.unique()
    for a in Date:
        if len(train_data_3.loc[a].index)>=8 or i=='Denmark' or i=='Netherlands':
            daily_sum = train_data_3.loc[a].sum()
        else:
            daily_sum = train_data_3.loc[a]
        ConfirmedCases.append(daily_sum.ConfirmedCases)
        Fatalities.append(daily_sum.Fatalities)
        X_1.append(loop)
        loop=loop+1
    train_data_final = pd.DataFrame({'ConfirmedCases':ConfirmedCases, 'Fatalities':Fatalities, 'X_1':X_1}, index=Date)
    len_x = len(Date)+1
    X_1 = np.array(list(range(1, len_x)))#.reshape(-1,1)
    X_2 = np.array(list(range(1, len_x))).reshape(-1,1)
    X_plot = Date
    y_1 = train_data_final["ConfirmedCases"].values#.reshape(-1,1)
    y_plot = train_data_final["ConfirmedCases"].values
    y_2 = train_data_final["Fatalities"].values.reshape(-1,1)
    
    #Prepare Test Data
    test_data_1 = test_data.loc[test_data["Country_Region"].isin([i])]
    test_data_2 = test_data_1.sort_values(by="Date")
    test_data_3 = test_data_2.set_index('Date')
    Date_test = test_data_1.Date.unique()
    
    X_test = test_data1.loc[test_data1["Country_Region"].isin([i])]
    X_train_map_index = X_test.index.values.min()
    X_train_map_index = np.datetime_as_string(X_train_map_index, unit='D')
    X_train_map_value = train_data_final.X_1.loc[X_train_map_index]
    X_train_map_value
    len_x_test = len(Date_test)
    X_test_1 = np.array(list(range(X_train_map_value, len_x_test+X_train_map_value)))
    X_test_2 = np.array(list(range(X_train_map_value, len_x_test+X_train_map_value))).reshape(-1,1)
    X_test_0 = np.array(list(range(5, 100))).reshape(-1,1)
    X_train_and_test = np.array(list(range(1, len_x_test+X_train_map_value))).reshape(-1,1)
    ForecastID = X_test["ForecastId"].values
    
    ####------ConfirmedCase-------> as y_1
    
    def func(x, a, b, c): # Logistic B equation 
        return a / (1.0 + numpy.power(x/b, c))
 
        
    # these are the same as the scipy defaults
    initialParameters = numpy.array([1.0, 1.0, 1.0])

    # curve fit the test data, ignoring warning due to initial parameter estimates
    warnings.filterwarnings("ignore")
    fittedParameters, pcov = curve_fit(func, X_1, y_1, initialParameters, maxfev=10000)

    modelPredictions = func(X_1, *fittedParameters) 

    absError = modelPredictions - y_1

    SE = numpy.square(absError) # squared errors
    MSE = numpy.mean(SE) # mean squared errors
    RMSE = numpy.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (numpy.var(absError) / numpy.var(y_1))

    print('Parameters:', fittedParameters)
    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)
    
    #Predict ConfirmedCases
    y_1_pred = modelPredictions = func(X_test_1, *fittedParameters)
    
    ####------Fatalities-------> as y_2
    
    # Fitting Linear Regression to Fatalities
    model = LinearRegression()
    model.fit(X_2, y_2)

    # Evaluate the testing error 
    score2 = model.score(X_2, y_2)
    
    #Predict ConfirmedCases
    y_2_pred = model.predict(X_test_2)
    
    # Visualizing the Polymonial Regression results
    def viz():
        plt.scatter(X_1, y_1, color='red')
        plt.plot(X_train_and_test, func(X_train_and_test, *fittedParameters), color='blue')
        plt.title(i)
        plt.xlabel('Date')
        plt.ylabel('ConfirmedCases')
        plt.show()
        return
    viz()
    
    #Prepare for Submission
    y_1 = np.squeeze(y_1_pred.transpose()).round()
    y_2 = np.squeeze(y_2_pred.transpose()).round()
    #map y_1, y_2 to each case of the date
    test_data_4 = test_data1.loc[test_data1["Country_Region"].isin([i])]
    test_data_4.ForecastId
    data_prep0 =pd.DataFrame({'ConfirmedCases':y_1, 'Fatalities':y_2},index=Date_test)
    data_prep1 = test_data_4.join(data_prep0)
    data_prep2 = data_prep1.sort_values(by='ForecastId')
    y_1_submit = data_prep2.ConfirmedCases.values    
    y_2_submit = data_prep2.Fatalities.values
    
    submit = pd.DataFrame({'ForecastId': ForecastID, 'ConfirmedCases': y_1_submit, 'Fatalities': y_2_submit}, index=ForecastID)#list(range(1, len_x_test+1)) )
    return submit
#i='Angola'
#i='Antigua and Barbuda'
#i='Australia'
i='Afghanistan'
#i='Netherlands'
score = train_logistic(i)
print(score) 
#Train Model for all countries
loop=0
countries = train_data["Country_Region"].unique()
for i in countries:
    print(i)
    submit = train_logistic(i)
    if loop ==0:
        output = submit.copy()
    else:
        output = pd.concat([output,submit])
    loop = loop +1
    print(submit)
print(output)
#Get Submission Data File
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")