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
import datetime as dt

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.patches as mpatches
ca_test_covid = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

ca_test_covid['Date']=pd.to_numeric(ca_test_covid.Date.str.replace('-',''))

ca_test_covid.head()

ca_train_covid = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

ca_train_covid.head()
#plt.close('all')

plt.figure()

ca_train_covid_confirm_cases=ca_train_covid[ca_train_covid['ConfirmedCases']>0]

ca_train_covid_confirm_cases['Date']=pd.to_numeric(ca_train_covid_confirm_cases.Date.str.replace('-',''))



def calculateDateRank(date_values):

    listRank=()

    arrayVal=np.array(date_values)

    rank=1

    for i in np.arange(len(arrayVal)):

        listRank=np.append(listRank,rank)

        rank=rank+1   

    

    return listRank
ca_train_covid_confirm_cases['DateRank']=calculateDateRank(ca_train_covid_confirm_cases.Date)

ca_train_covid_confirm_cases.head()
ca_test_covid['DateRank']=calculateDateRank(ca_test_covid.Date)

ca_train_covid_confirm_cases.plot.scatter(x='Date',y='ConfirmedCases')

ca_train_covid_confirm_cases.plot.scatter(x='Date',y='Fatalities')
ca_train_covid_confirm_cases.corr()
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score,mean_squared_log_error

from sklearn.preprocessing import PolynomialFeatures
# Create linear regression model

#regModelConfirmedCases = linear_model.LinearRegression()

regModelConfirmedCases = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))

trainX=np.array(ca_train_covid_confirm_cases['DateRank']).reshape(-1,1)

trainY=np.array(ca_train_covid_confirm_cases['ConfirmedCases']).reshape(-1,1)
# Train the model using the training sets

poly_features = PolynomialFeatures(degree=4)



X_train_poly = poly_features.fit_transform(trainX)    

    

regModelConfirmedCases.fit(X_train_poly, trainY)
# Make predictions using the testing set

ca_test_covid_date_gt_26Mar=ca_test_covid

testX_Date=np.array(ca_test_covid.DateRank).reshape(-1,1)

pred_ConfirmedCase = regModelConfirmedCases.predict(poly_features.fit_transform(testX_Date))
#print("Graph of a Train Date point vs ConfirmedCase prediction regression line")

# Plot outputs

plt.scatter(trainX, trainY,  color='green')

plt.plot(testX_Date, pred_ConfirmedCase, color='blue', linewidth=1)

plt.scatter(testX_Date, pred_ConfirmedCase,  color='yellow')



plt.title('training / predicted values across regression line for ')

plt.xticks(())

plt.yticks(())

plt.xlabel("Date Rank")

plt.ylabel("ConfirmedCase")



green_patch = mpatches.Patch(color='green', label='Train Values')

yellow_patch = mpatches.Patch(color='yellow', label='Predicted Values')



plt.legend(handles=[green_patch,yellow_patch])



plt.show()
# Train the model using the training sets

trainY_Fatalities=np.array(ca_train_covid_confirm_cases['Fatalities']).reshape(-1,1)

regrModelFatalities = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))

regrModelFatalities.fit(X_train_poly, trainY_Fatalities)



predY_Fatalities=regrModelFatalities.predict(poly_features.fit_transform(testX_Date))

# Plot outputs



plt.scatter(trainX, trainY_Fatalities,  color='green')

plt.plot(testX_Date, predY_Fatalities, color='blue', linewidth=1)

plt.scatter(testX_Date, predY_Fatalities,  color='yellow')



plt.xticks(())

plt.yticks(())

plt.xlabel("Date Rank")

plt.ylabel("Fatalities")

plt.title('training data (DateRank)/Predicted Data(Fatalities) and regression line for ')



green_patch = mpatches.Patch(color='green', label='Train Values')

yellow_patch = mpatches.Patch(color='yellow', label='Predicted Values')



plt.legend(handles=[green_patch,yellow_patch])

plt.show()

#print(ca_test_covid.Date)
ca_test_covid['ConfirmedCases']=pred_ConfirmedCase

ca_test_covid['Fatalities']=predY_Fatalities

ca_test_covid_submission=ca_test_covid[['ForecastId','ConfirmedCases','Fatalities']]

ca_test_covid_submission.to_csv('submission.csv', index=False)

ca_test_covid_submission