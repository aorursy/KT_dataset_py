# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from datetime import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv(os.path.join(dirname,filenames[2]))

# data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data.sort_values('Confirmed', ascending=False).head()
data.sort_values('Deaths', ascending=False).head()
data.groupby('Country/Region').sum().sort_values('Deaths', ascending=False).head(20)

data_group = data.groupby('Country/Region').sum()
data[data['Country/Region'] == 'South Korea'].sort_values('Deaths', ascending=False)
data_country = data.pivot_table(values='Confirmed', index='ObservationDate', columns='Country/Region', aggfunc='sum')

data_country.fillna(0)
data[data['Country/Region'] == 'South Korea']
data_country.plot(figsize=(20,5), legend=False)

plt.ylabel('Confirmed')

plt.xlabel('Date')

plt.title('Corona19')
data
today_data = data[data['ObservationDate'] == '03/11/20']
today_data.groupby('Country/Region').sum().sort_values('Confirmed', ascending=False).head(5)
top5_list = list((today_data.groupby('Country/Region').sum().sort_values('Confirmed', ascending=False).head(5)).index)
top5_list
top5 = data[data['Country/Region'].isin(top5_list)]
top5
top5_pivot = top5.pivot_table(values='Confirmed', index='ObservationDate', columns='Country/Region', aggfunc='sum')

top5_pivot.fillna(0)
top5_pivot.plot(figsize=(8,5))

plt.ylabel('Confirmed')

plt.xlabel('Date')

plt.title('Corona19')
korea = data[data['Country/Region'] == 'South Korea']
korea.plot(x='ObservationDate', y='Confirmed')
korea = korea.reset_index(drop=True)
korea['Attacked'] = np.nan

korea
for i in range(0,len(korea)):

    if i == 0:

        korea['Attacked'][0] = korea['Confirmed'][0]

    else:

        korea['Attacked'][i] = korea['Confirmed'][i] - korea['Confirmed'][i-1]
korea.plot(x='ObservationDate', y='Attacked')
from sklearn.linear_model import LinearRegression
type(korea['ObservationDate'][0])
korea['date'] = np.nan
for i in range(len(korea)):

    if len(korea['ObservationDate'][i]) != 8:

        korea['ObservationDate'][i] = korea['ObservationDate'][i][0:8]
korea
for i in range(len(korea)):

    korea['date'][i] = datetime.strptime(korea['ObservationDate'][i], "%m/%d/%y")

#     korea['date'][i] = korea['date'][i].strftime('%Y%m%d')
y = korea['Attacked']

X = korea.index

line_fitter = LinearRegression()

line_fitter.fit(X.values.reshape(-1, 1), y)
line_fitter.intercept_ #절편

line_fitter.coef_ #기울기
line_fitter.predict([[len(korea)+1]])
plt.figure(figsize=(20,5))

plt.plot(X, y)

plt.plot(X,line_fitter.predict(X.values.reshape(-1,1)))

plt.xticks(rotation=60, fontsize=10)

plt.show()