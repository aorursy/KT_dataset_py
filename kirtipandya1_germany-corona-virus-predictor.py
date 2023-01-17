# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import AutoDateLocator
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
#df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', index_col='Date', parse_dates=True)
df = pd.read_csv('../input/coronavirus-2019ncov/covid-19-all.csv', index_col='Date', parse_dates=True)
#df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df[['Confirmed','Recovered','Deaths']] = df[['Confirmed','Recovered','Deaths']].fillna(0).astype(int)
df_china = df[:][df['Country/Region']=='China']
df_china['Day Number'] = range(1,df_china.shape[0]+1)

plt.style.use('fivethirtyeight')
#plt.plot(df_china['Day Number'], df_china['Confirmed'],color='b',marker='.', label="Number of cases")
#plt.plot(df_china['Day Number'], df_china['Recovered'],color='g',marker='.', label="Recovered",linestyle='--')
#plt.plot(df_china['Day Number'], df_china['Deaths'],color='r',marker='.', label="Death",linestyle='--')

df_china[['Confirmed','Recovered', 'Deaths']].plot(color=['b','g','r'])

plt.xlabel("Date")
plt.xticks(rotation=90)
plt.ylabel("Cases")
plt.title("Corona virus in China")
plt.legend()
plt.grid(False)
#plt.tight_layout()
plt.show()
#df_china.tail(33)
df_china_temp = df_china.tail(33)
active = sum(df_china_temp['Confirmed']) - sum(df_china_temp['Recovered']) - sum(df_china_temp['Deaths'])
labels = ['Active','Recovered','Deaths']
values = [active, sum(df_china_temp['Recovered']), sum(df_china_temp['Deaths'])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)],layout=go.Layout(title='China Corona Virus Cases'))
fig.show()
df_italy = df[:][df['Country/Region']=='Italy']
df_italy['Day Number'] = range(1,df_italy.shape[0]+1)

plt.style.use('fivethirtyeight')
#plt.plot(df_china['Day Number'], df_china['Confirmed'],color='b',marker='.', label="Number of cases")
#plt.plot(df_china['Day Number'], df_china['Recovered'],color='g',marker='.', label="Recovered",linestyle='--')
#plt.plot(df_china['Day Number'], df_china['Deaths'],color='r',marker='.', label="Death",linestyle='--')

df_italy[['Confirmed','Recovered', 'Deaths']].plot(color=['b','g','r'])

plt.xlabel("Date")
plt.xticks(rotation=90)
plt.ylabel("Cases")
plt.title("Corona virus in Italy")
plt.legend()
plt.grid(True)
#plt.tight_layout()
plt.show()

df_italy['Active'] = df_italy['Confirmed'] - df_italy['Recovered'] - df_italy['Deaths']
labels = ['Active','Recovered','Deaths']
values = [df_italy['Active'][-1:].iloc[-1], df_italy['Recovered'][-1:].iloc[-1], df_italy['Deaths'][-1:].iloc[-1]]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)],layout=go.Layout(title='Italy Corona Virus Cases'))
fig.show()


confirmed_cases = df_italy['Confirmed'].as_matrix()
added_cases = np.diff(confirmed_cases)
added_cases_no_day = range(1,added_cases.shape[0]+1)
df_italy['Added Cases'] = 0
df_italy['Added Cases'][1:] = added_cases

plt.style.use('fivethirtyeight')
#plt.plot(added_cases_no_day, added_cases,color='b',marker='o', label="New Added Cases")
df_italy[['Added Cases']].plot(color='b',marker='o', label="New Added Cases")

plt.xlabel("Date")
plt.ylabel("New Added Cases")
plt.title("Corona virus in Italy",)
plt.legend()
plt.grid(True)
#plt.tight_layout()
plt.show()
df_germany = df[:][df['Country/Region']=='Germany']
df_germany['Day Number'] = range(1,df_germany.shape[0]+1)
df_germany.to_csv('Dataframe_Germany.csv')
df_germany.tail()
df_germany['Active'] = df_germany['Confirmed'] - df_germany['Recovered'] - df_germany['Deaths']
labels = ['Active','Recovered','Deaths']
values = [df_germany['Active'][-1:].iloc[-1], df_germany['Recovered'][-1:].iloc[-1], df_germany['Deaths'][-1:].iloc[-1]]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)],layout=go.Layout(title='Germany Corona Virus Cases'))
fig.show()
plt.style.use('fivethirtyeight')

#plt.plot(df_germany['Day Number'], df_germany['Confirmed'],color='b',marker='D', label="Number of cases")
#plt.plot(df_germany['Day Number'], df_germany['Recovered'],color='g',marker='.', label="Recovered",linestyle='--')
#plt.plot(df_germany['Day Number'], df_germany['Deaths'],color='r',marker='.', label="Death",linestyle='--')

df_germany[['Confirmed','Recovered', 'Deaths']].plot(color=['b','g','r'],marker='>' )

plt.xlabel("Date")
#plt.xticks(rotation=45)
plt.ylabel("Cases")
plt.title("Corona virus in Germany",)
plt.legend()
plt.grid(True)
#plt.tight_layout()
plt.show()
#df_germany.set_index('Date', inplace=True)

plt.style.use('fivethirtyeight')
#plt.plot(df_germany['Day Number'], df_germany['Recovered'],color='g',marker='.', label="Recovered",linestyle='--')
#plt.plot(df_germany['Day Number'], df_germany['Deaths'],color='r',marker='.', label="Death",linestyle='--')

df_germany[['Recovered', 'Deaths']].plot(marker='>')

plt.xlabel("Date")
plt.xticks(rotation=90)
plt.ylabel("Cases")
plt.title("Corona virus in Germany",)
plt.legend()
plt.grid(True)
#plt.tight_layout()
plt.show()

modal_germany = LinearRegression(fit_intercept=True)
poly = PolynomialFeatures(degree=12)
num_days_poly = poly.fit_transform(df_germany['Day Number'].as_matrix().reshape(-1,1))
poly_reg = modal_germany.fit(num_days_poly, df_germany['Confirmed'].as_matrix().reshape(-1,1))
predictions_for_given_days = modal_germany.predict(num_days_poly)
#print("coef_ :",modal_germany.coef_,"intercept_:",modal_germany.intercept_)
plt.style.use('fivethirtyeight')
plt.plot(df_germany['Day Number'], df_germany['Confirmed'],color='b',marker='D', label="Number of cases")
plt.plot(df_germany['Day Number'], predictions_for_given_days,color='k',marker='o', label="Prediction",linestyle='--')

plt.plot(df_germany['Day Number'], df_germany['Recovered'],color='g',marker='.', label="Recovered",linestyle='--')
plt.plot(df_germany['Day Number'], df_germany['Deaths'],color='r',marker='.', label="Death",linestyle='--')
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("Corona virus in Germany",)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
tomorrow_value = df_germany["Day Number"].iloc[-1] + 1 
value_prediction = poly.fit_transform(np.array([[tomorrow_value]]))
prediction = modal_germany.predict(value_prediction)
#print(f'Prediction for tomorrow (for day number {tomorrow_value}) : {prediction} cases ')
confirmed_cases = df_germany['Confirmed'].as_matrix()
added_cases = np.diff(confirmed_cases)
added_cases_no_day = range(1,added_cases.shape[0]+1)
df_germany['Added Cases'] = 0
df_germany['Added Cases'][1:] = added_cases
added_cases_prediction = prediction[0][0] - df_germany['Confirmed'][-1:]
#print(f'Hospitals should be ready with {round(added_cases_prediction[:1].iloc[-1])} new beds.')
df_germany[['Added Cases']].tail(10)



plt.style.use('fivethirtyeight')
#plt.plot(added_cases_no_day, added_cases,color='b',marker='o', label="New Added Cases")
df_germany[['Added Cases']].plot(color='b',marker='o', label="New Added Cases")

plt.xlabel("Date")
plt.ylabel("New Added Cases")
plt.title("Corona virus in Germany",)
plt.legend()
plt.grid(True)
#plt.tight_layout()
plt.show()
df_temp2=pd.DataFrame()
df_temp2['Confirmed'] = df_germany['Confirmed']
df_temp2['Deaths'] = df_germany['Deaths']
df_temp2['Recovered'] = df_germany['Recovered']
sns.pairplot(df_temp2)

