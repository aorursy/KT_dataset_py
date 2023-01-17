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
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import plotly.graph_objects as go

import seaborn as sns
#df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df = pd.read_csv('../input/coronavirus-2019ncov/covid-19-all.csv')

df['Date'] = pd.to_datetime(df['Date'])

df[['Confirmed','Recovered','Deaths']] = df[['Confirmed','Recovered','Deaths']].fillna(0).astype(int)
df_china = df[:][df['Country/Region']=='Mainland China']

df_china['Day Number'] = range(1,df_china.shape[0]+1)



plt.style.use('fivethirtyeight')

plt.plot(df_china['Day Number'], df_china['Confirmed'],color='b',marker='.', label="Number of cases")

plt.plot(df_china['Day Number'], df_china['Recovered'],color='g',marker='.', label="Recovered",linestyle='--')

plt.plot(df_china['Day Number'], df_china['Deaths'],color='r',marker='.', label="Death",linestyle='--')

plt.xlabel("Day Number")

plt.xticks(rotation=45)

plt.ylabel("Cases")

plt.title("Corona virus in China")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
df_italy = df[:][df['Country/Region']=='Italy']

df_italy['Day Number'] = range(1,df_italy.shape[0]+1)

df_italy.tail()
df_italy['Active'] = df_italy['Confirmed'] - df_italy['Recovered'] - df_italy['Deaths']

labels = ['Active','Recovered','Deaths']

values = [df_italy['Active'][-1:].iloc[-1], df_italy['Recovered'][-1:].iloc[-1], df_italy['Deaths'][-1:].iloc[-1]]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)],layout=go.Layout(title='Italy Corona Virus Cases'))

fig.show()
plt.style.use('fivethirtyeight')

plt.plot(df_italy['Day Number'], df_italy['Confirmed'],color='b',marker='D', label="Number of cases")

plt.plot(df_italy['Day Number'], df_italy['Recovered'],color='g',marker='.', label="Recovered",linestyle='--')

plt.plot(df_italy['Day Number'], df_italy['Deaths'],color='r',marker='.', label="Death",linestyle='--')

plt.xlabel("Day Number")

#plt.xticks(rotation=45)

plt.ylabel("Cases")

plt.title("Corona virus in Italy",)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
plt.style.use('fivethirtyeight')

#plt.plot(df_italy['Day Number'], df_italy['Confirmed'],color='b',marker='D', label="Number of cases")

plt.plot(df_italy['Day Number'], df_italy['Recovered'],color='g',marker='.', label="Recovered",linestyle='--')

plt.plot(df_italy['Day Number'], df_italy['Deaths'],color='r',marker='.', label="Death",linestyle='--')

plt.xlabel("Day Number")

#plt.xticks(rotation=45)

plt.ylabel("Cases")

plt.title("Corona virus in Italy",)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

modal_italy = LinearRegression(fit_intercept=True)

poly = PolynomialFeatures(degree=8)

num_days_poly = poly.fit_transform(df_italy['Day Number'].as_matrix().reshape(-1,1))

poly_reg = modal_italy.fit(num_days_poly, df_italy['Confirmed'].as_matrix().reshape(-1,1))

predictions_for_given_days = modal_italy.predict(num_days_poly)

print("modal_italy.coef_ :",modal_italy.coef_,"modal_italy.intercept_:",modal_italy.intercept_)
plt.style.use('fivethirtyeight')

plt.plot(df_italy['Day Number'], df_italy['Confirmed'],color='b',marker='D', label="Number of cases")

plt.plot(df_italy['Day Number'], predictions_for_given_days,color='k',marker='o', label="Prediction",linestyle='--')



plt.plot(df_italy['Day Number'], df_italy['Recovered'],color='g',marker='.', label="Recovered",linestyle='--')

plt.plot(df_italy['Day Number'], df_italy['Deaths'],color='r',marker='.', label="Death",linestyle='--')

plt.xlabel("Day Number")

plt.ylabel("Cases")

plt.title("Corona virus in Italy",)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
tomorrow_value = df_italy["Day Number"].iloc[-1] + 1 

value_prediction = poly.fit_transform(np.array([[tomorrow_value]]))

prediction = modal_italy.predict(value_prediction)

print(f'Prediction for tomorrow (for day number {tomorrow_value}) : {prediction} cases ')
modal_italy = LinearRegression(fit_intercept=True)

poly = PolynomialFeatures(degree=8)

num_days_poly = poly.fit_transform(df_italy['Day Number'].as_matrix().reshape(-1,1))

poly_reg = modal_italy.fit(num_days_poly, df_italy['Deaths'].as_matrix().reshape(-1,1))

predictions_for_given_days = modal_italy.predict(num_days_poly)

print("modal_italy.coef_ :",modal_italy.coef_,"modal_italy.intercept_:",modal_italy.intercept_)
tomorrow_value = df_italy["Day Number"].iloc[-1] + 1 

value_prediction = poly.fit_transform(np.array([[tomorrow_value]]))

prediction = modal_italy.predict(value_prediction)

print(f'Prediction for tomorrow (for day number {tomorrow_value}) : {prediction} deaths ')
confirmed_cases = df_italy['Confirmed'].as_matrix()

added_cases = np.diff(confirmed_cases)

added_cases_no_day = range(1,added_cases.shape[0]+1)



plt.style.use('fivethirtyeight')

plt.plot(added_cases_no_day, added_cases,color='b',marker='o', label="New Added Cases")

plt.xlabel("Day Number")

plt.ylabel("New Added Cases")

plt.title("Corona virus in Italy",)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
df_italy['Added Cases'] = 0

df_italy['Added Cases'][1:] = added_cases

added_cases_prediction = prediction[0][0] - df_italy['Confirmed'][-1:]

print(f'Hospitals should be ready with {round(added_cases_prediction[:1].iloc[-1])} new beds.')

df_italy[['Date','Added Cases']].tail()

df_temp2=pd.DataFrame(df_italy["Date"])

df_temp2['Confirmed'] = df_italy['Confirmed']

df_temp2['Deaths'] = df_italy['Deaths']

df_temp2['Recovered'] = df_italy['Recovered']

sns.pairplot(df_temp2)