# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import time

from datetime import datetime

import warnings

warnings.filterwarnings('ignore')

import matplotlib.dates as mdates

%matplotlib inline

from sklearn.linear_model import LinearRegression

%matplotlib inline

import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')
data = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv', usecols=['Date','ConfirmedCases','Fatalities'], parse_dates=['Date'])
df1 = data[data['ConfirmedCases']>0.0]
ncov_CA = df1

ncov_CA = pd.DataFrame(ncov_CA.groupby(['Date'])['ConfirmedCases','Fatalities'].sum()).reset_index()

fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=ncov_CA['Date'], y=ncov_CA['ConfirmedCases'], name='ConfirmedCases'))

fig1.add_trace(go.Scatter(x=ncov_CA[21:23]['Date'], y=ncov_CA[21:23]['ConfirmedCases'], mode='markers', name='Inflection', marker=dict(color='Red',line=dict(width=5, color='Red'))))

fig1.layout.update(title_text='Pandemic Growth in California',xaxis_showgrid=False, yaxis_showgrid=False, width=800,

        height=500,font=dict(

#         family="Courier New, monospace",

        size=12,

        color="white"

    ))

fig1.layout.plot_bgcolor = 'Black'

fig1.layout.paper_bgcolor = 'Black'

fig1.show()

df1.set_index('Date',inplace=True)
#plot data

fig, ax = plt.subplots(figsize=(10,5))

df1.plot(ax=ax)

#set ticks every week

ax.xaxis.set_major_locator(mdates.WeekdayLocator())

#set major ticks format

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")
col = ['Id', 'Lat', 'Long','ConfirmedCases', 'Fatalities','Date']

df_final = train[col]

df_final = df_final.reset_index()

df_final
df_final['Date'] = pd.to_numeric(df_final.Date.str.replace('-',''))

print( df_final )
col_X = ['Date']

col_Ycon = ['ConfirmedCases']

col_Yfat = ['Fatalities']

trainX = df_final[col_X].iloc[:,:]

trainYcon = df_final[col_Ycon].iloc[:,:]

trainF= df_final[col_Yfat].iloc[:,:]
import numpy

import matplotlib.pyplot as plt 

import seaborn as sns 

sns.set()

import numpy as np

def polyfit(x, y,degree):

       

        results = {}



        coeffs = np.polyfit(x, y, degree)



         # Polynomial Coefficients

        results['polynomial'] = coeffs.tolist()



        # r-squared

        p = np.poly1d(coeffs)

        # fit values, and mean

        yhat = p(x)                         # or [p(z) for z in x]

        ybar = np.sum(y)/len(y)          # or sum(y)/len(y)

        ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])

        sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])

        results['R square:'] = ssreg / sstot

        print('equation :')

        print( p)

        return p, results
p, results = polyfit(trainX['Date'].values,trainYcon['ConfirmedCases'].values, degree = 2)
plt.figure(figsize=(20,5))

plt.plot(trainX['Date'].values,trainYcon['ConfirmedCases'].values, 'o', label='original values ')# original values 

plt.plot(trainX['Date'].values,p(trainX['Date']), '-o', label='predicted values')# predicted values 

plt.legend()

plt.tight_layout()

plt.xticks(rotation = 45)

plt.show()
test['Date'] = pd.to_numeric(test.Date.str.replace('-',''))

test.head() 
test['confirmed cases'] = p(test['Date'])
p, results = polyfit(trainX['Date'].values,trainF['Fatalities'].values, degree = 2)
plt.figure(figsize=(20,5))

plt.plot(trainX['Date'].values,trainF['Fatalities'].values, 'o', label='original values ')# original values 

plt.plot(trainX['Date'].values,p(trainX['Date']), '-o', label='predicted values')# predicted values 

plt.legend()

plt.tight_layout()

plt.xticks(rotation = 45)

plt.show()

test['Fatalities'] = p(test['Date'])

test['Fatalities'] = test['Fatalities']
submission = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')

submission['ConfirmedCases']  = test['confirmed cases']

submission['Fatalities']  = test['Fatalities']

submission.head()