import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) #Đọc file
import os
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv').fillna('-')
tempTrain = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv').fillna('-')
tempTest = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv').fillna('-')
submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv').fillna('-')


import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px # install plotly
from datetime import datetime
%matplotlib inline
def drawPie(dataFrame, indexValue, label, title="Default"):
    fig = px.pie(train, values=indexValue, names=label, title=title)
    fig.update_traces(textposition='inside')
    fig.show()
def drawPie(dataFrame, indexValue, label, title="Default"):
    fig = px.pie(train, values=indexValue, names=label, title=title)
    fig.update_traces(textposition='inside')
    fig.show()
    
getTopList = 15
grouped_multiple = train.groupby(['Country_Region'], as_index=False)['TargetValue'].sum()
countryTop = grouped_multiple.nlargest(getTopList, 'TargetValue')['Country_Region']
newList = train[train['Country_Region'].isin(countryTop.values)]
line = newList.groupby(['Date', 'Country_Region'], as_index=False)['TargetValue'].sum()
line = line[line['TargetValue'] >= 0]

line.pivot(index="Date", columns="Country_Region", values="TargetValue").plot(figsize=(10,5))
plt.grid(zorder=0)
plt.title('Top ' + str(getTopList) +' ConfirmedCases & Fatalities', fontsize=18, pad=10)
plt.ylabel('People')
plt.xlabel('Date')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
drawPie(train, 'TargetValue', 'Target', 'Summary ConfirmedCases & Fatalities')
redate = pd.to_datetime(tempTrain['Date'], errors='coerce')
tempTrain['Date']= redate.dt.strftime("%Y%m%d").astype(int)
targets = train['Target'].unique()
for index in range(0, len(targets)):
    tempTrain['Target'].replace(targets[index], index, inplace=True)
    
feature_cols = ['Population', 'Weight', 'Date', 'Target']
X = tempTrain[feature_cols]
y = tempTrain['TargetValue']
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)


model = RandomForestRegressor(n_jobs=-1, n_estimators = 50)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Score: "+ str(score))
