import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



path = "/kaggle/input/covid19-global-forecasting-week-4/"

pred = pd.read_csv(path+"train.csv")

test = pd.read_csv(path+"test.csv")



print(pred)



from datetime import datetime

train_final_x = []

for i in range(0,len(pred)):

    dt = pred['Date'][i]

    dt = datetime.strptime(dt, '%Y-%m-%d')

    dt = dt.timestamp()%150000000

    dt = dt/100

    pred['Date'][i] = dt



print(pred['Date'])



le=LabelEncoder()

pred['Country_Region'] = le.fit_transform(pred['Country_Region'])

pred['Province_State'] = pred['Province_State'].replace(np.nan,'aawa')

pred['Province_State'] = le.fit_transform(pred['Province_State'])
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import tree

from sklearn import svm

from sklearn.metrics import f1_score

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression



rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

rf1 = RandomForestRegressor(n_estimators = 1000, random_state = 42)



X = pred[['Province_State','Country_Region','Date']]

y = pred['ConfirmedCases']

y2 = pred['Fatalities']



rf.fit(X, y)

rf1.fit(X,y2)



trees = tree.DecisionTreeClassifier()

trees1 = tree.DecisionTreeClassifier()

trees.fit(X,y)

trees1.fit(X,y2)
for i in range(0,len(test)):

    dt = test['Date'][i]

    dt = datetime.strptime(dt, '%Y-%m-%d')

    dt = dt.timestamp()%150000000

    dt = dt/100

    test['Date'][i] = dt



test['Country_Region'] = le.fit_transform(test['Country_Region'])

test['Province_State'] = test['Province_State'].replace(np.nan,'aawa')

test['Province_State'] = le.fit_transform(test['Province_State'])



x_test = test[['Province_State','Country_Region','Date']]

y_pred = trees.predict(x_test)



deaths = rf1.predict(x_test)



import csv

row_list = [["ForecastId","ConfirmedCases","Fatalities"]]

for i in range(0,len(test)):

    temp = [test['ForecastId'][i],y_pred[i],deaths[i]]

    row_list.append(temp)

print(row_list)



with open('submission.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerows(row_list)