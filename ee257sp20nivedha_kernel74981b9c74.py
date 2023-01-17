# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")
train.head()
train.shape
import plotly.express as px
tot_confirmed = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})
tot_fatalities = train.groupby(['Date']).agg({'Fatalities':['sum']})
tot_case_bydate = tot_confirmed.join(tot_fatalities)
tot_case_bydate.reset_index(inplace = True)
# Later need to put into one figure
fig = px.scatter(tot_case_bydate, x = 'Date', y = 'ConfirmedCases',
                width=800, height=400)
fig.show()
train['Province_State'].fillna("",inplace = True)
test['Province_State'].fillna("",inplace = True)
# Drop Province State as it has no use.
train['Country_Region'] = train['Country_Region'] + ' ' + train['Province_State']
test['Country_Region'] = test['Country_Region'] + ' ' + test['Province_State']
train.drop("Province_State", axis=1, inplace=True)
train.head()
del test['Province_State']
test.head()
def split_date(date):
    date = date.split('-')
    date[0] = int(date[0])
    if(date[1][0] == '0'):
        date[1] = int(date[1][1])
    else:
        date[1] = int(date[1])
    if(date[2][0] == '0'):
        date[2] = int(date[2][1])
    else:
        date[2] = int(date[2])    
    return date
train.Date = train.Date.apply(split_date)
test.Date = test.Date.apply(split_date)
year = []
month = []
day = []
for i in train.Date:
    year.append(i[0])
    month.append(i[1])
    day.append(i[2])
train['Year'] = year
train['Month'] = month
train['Day'] = day
del train['Date']   
year = []
month = []
day = []
for i in test.Date:
    year.append(i[0])
    month.append(i[1])
    day.append(i[2])
test['Year'] = year
test['Month'] = month
test['Day'] = day
del test['Date']
del train['Year']
del test['Year']
del train['Id']
del test['ForecastId']
train.head()
test.head()
train['ConfirmedCases'] = train['ConfirmedCases'].apply(int)
train['Fatalities'] = train['Fatalities'].apply(int)
cases = train.ConfirmedCases
fatalities = train.Fatalities
del train['ConfirmedCases']
del train['Fatalities']
from sklearn.preprocessing import LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler
lb = LabelEncoder()
train['Country_Region'] = lb.fit_transform(train['Country_Region'])
test['Country_Region'] = lb.transform(test['Country_Region'])
scaler = MinMaxScaler()
x_train = scaler.fit_transform(train.values)
x_test = scaler.transform(test.values)
#XGBRegressor model for cases
from xgboost import XGBRegressor
rf = XGBRegressor(n_estimators = 1500 , max_depth = 15, learning_rate=0.1)
rf.fit(x_train,cases)
cases_pred = rf.predict(x_test)
cases_pred
cases_pred = np.around(cases_pred,decimals = 0)
cases_pred 
#XGBRegressor model for fatalities
rf = XGBRegressor(n_estimators = 1500 , max_depth = 15, learning_rate=0.1)
rf.fit(x_train,fatalities)
fatalities_pred = rf.predict(x_test)
fatalities_pred
fatalities_pred = np.around(fatalities_pred,decimals = 0)
fatalities_pred
submission['ConfirmedCases'] = cases_pred
submission['Fatalities'] = fatalities_pred
submission.head()
submission.to_csv("submission.csv" , index = False)
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
X_train, X_test, y_train, y_test = train_test_split(train, cases, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')