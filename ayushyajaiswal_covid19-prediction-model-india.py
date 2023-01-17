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
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.express as px

import plotly.graph_objects as go

import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
ind_state = pd.read_excel('/kaggle/input/India_covid19_states_wise.xlsx', index_col=0)
ind_daily = pd.read_excel('/kaggle/input/per_day_cases.xlsx')
ind_complete = pd.read_csv('https://raw.githubusercontent.com/covid19india/api/gh-pages/csv/latest/raw_data.csv', index_col = 0)
ind_state.info()
ind_daily.info()
ind_complete.info()
ind_state.head()
ind_daily.head()
ind_complete.head()
ind_complete.drop(['Estimated Onset Date', 'Notes', 'Contracted from which Patient (Suspected)', 'Source_1', 'Source_2', 'Source_3'], axis = 1, inplace = True)
ind_complete.head()
plt.figure(figsize = (12,8))

sns.heatmap(ind_complete.isnull(), yticklabels=False, cbar = False, cmap='viridis')
ind_complete.drop(['State Patient Number', 'Detected City', 'Detected District', 'Nationality', 'Type of transmission', 'Backup Notes'], axis = 1, inplace = True)
ind_complete.head()
total = ind_daily.iloc[-1,1]

print("Total number of Confirmed cases till 26th April 2020 in India: {}".format(total))
ind_state.drop('Cured / Discharged', axis = 1).style.background_gradient(cmap = 'Reds')
active = ind_daily.iloc[-1,3]

cured = sum(ind_daily['Daily Recovery'])

deaths = ind_daily.iloc[-1,4]

print("Total number of Currently Active cases in India: {}".format(active))

print("Total number of Cured/Discharged cases in India: {}".format(cured))

print("Total number of Deaths in India: {}".format(deaths))
Tot_cases = ind_state.groupby('Name of State / UT')['Active Cases'].sum().sort_values(ascending = False).to_frame()

Tot_cases.style.background_gradient(cmap = 'Reds')
plt.figure(figsize = (18,35))

data = ind_state.copy()

data.sort_values('Total Confirmed Cases', ascending = False, inplace = True)



#sns.set_color_codes('pastel')

sns.barplot(y = 'Name of State / UT', x = 'Total Confirmed Cases', data = data, label = 'Total', color = 'r', saturation= 20)



sns.barplot(x = 'Active Cases', y = 'Name of State / UT', data = data, label = 'Active', color = 'b')



sns.barplot(x = 'Cured / Discharged', y = 'Name of State / UT', data = data, label = 'Cured', color = 'g')



plt.legend(loc = 'lower right', ncol = 3, frameon = True)
df = px.data.tips()

fig = px.bar(ind_state.sort_values(by = 'Total Confirmed Cases'), x = 'Name of State / UT', 

             y = 'Total Confirmed Cases', height = 700, width = 900, 

             title = 'Total COVID19 Cases per State', color='Total Confirmed Cases', 

             hover_data=['Total Confirmed Cases', 'Active Cases', 'Cured / Discharged', 'Deaths'],

             color_continuous_scale='Bluered_r')

fig.show()
fig = px.scatter(ind_daily, x = 'Date', y = 'Total Cases', color = 'Total Cases', color_continuous_scale='Bluered_r',

                title = 'Total COVID19 Cases in India', height = 700, width = 900)

fig.show()
fig = px.bar(ind_daily, x = 'Date', y = 'Daily New Cases', title = 'Daily New Cases in India', color = 'Daily New Cases',

            color_continuous_scale='Bluered_r', height = 700, width = 900)

fig.show()
fig = px.scatter(ind_daily, x = 'Date', y = 'Active Cases', color = 'Active Cases', color_continuous_scale='Bluered_r', 

                title = 'Active Cases in India', height = 700, width = 900)

fig.show()
fig = px.scatter(ind_daily, x = 'Date', y = 'Total Deaths', color = 'Total Deaths', color_continuous_scale='Bluered_r',

                title = 'Total Deaths', height = 700, width = 900)

fig.show()
fig = px.bar(ind_daily, x = 'Date', y = 'Daily Deaths', color = 'Daily Deaths', color_continuous_scale='Bluered_r',

            title = 'Daily Deaths', width = 900, height = 700)

fig.show()
ind_daily_c = ind_daily.copy()

ind_daily_c.set_index('Date', inplace = True)

ind_daily_c.drop(['Total Cases', 'Active Cases', 'Total Deaths'], axis = 1).iplot(mode = 'markers+lines', size = 5,

                                            yTitle = 'Daily new Coronavirus Cases + Cured + Deaths', title = 'New Cases vs New Recoveries vs New Deaths')
ind_complete.columns
temp_age = pd.DataFrame(ind_complete['Age Bracket'])

temp_gender = pd.DataFrame(ind_complete['Gender'])
temp_age.dropna(inplace = True)

temp_gender.dropna(inplace = True)
temp_age['Count'] = 1

temp_gender['Count'] = 1
total = sum(temp_gender['Count'])

fig = px.histogram(temp_gender, x = 'Gender', y = 'Count', 

                   title = 'Sample size: {} patients'.format(total), color = 'Gender', height = 700)

fig.show()
temp_age[temp_age['Age Bracket'] == '28-35']
temp_age['Age Bracket'][925] = 31

temp_age['Age Bracket'][926] = 31

temp_age['Age Bracket'][927] = 31

temp_age['Age Bracket'][928] = 31
temp_age[temp_age['Age Bracket'] == '28-35']
temp_age.dtypes
temp_age = temp_age.astype({'Age Bracket': 'float64'})
temp_age.dtypes
age_sum = [0,0,0,0,0,0,0,0,0,0]

j = 0

for i in range(0, 100, 10):

    c = temp_age[(temp_age['Age Bracket'] < (i+11)) & (temp_age['Age Bracket'] > i)].count()

    age_sum[j] = c['Count']

    j += 1
age_group = ['0 - 10', '11-20', '21 - 30', '31 - 40', '41 - 50', '51 - 60', '61 - 70', '71 - 80', '81 - 90', '91 - 100']
df = pd.DataFrame(data = age_sum, columns = ['Count'])
df['Age Group'] = age_group
df
total = sum(df['Count'])

fig = px.bar(df, y = 'Count', x = 'Age Group', color = 'Age Group', title = 'Sample Size: {} patients'.format(total), height = 700, width = 900)

fig.show()
from fbprophet import Prophet
fig = px.bar(ind_daily, x = 'Date', y = 'Total Cases', title = 'Daily Growth in number', height = 700, width = 900)

fig.show()
Total_cases = ind_daily[['Date', 'Total Cases']]
Total_cases.head()
Total_cases.columns = ['ds', 'y']

Total_cases['ds'] = pd.to_datetime(Total_cases['ds'])

Total_cases.head()
m = Prophet(interval_width=0.95)
m.fit(Total_cases)
future = m.make_future_dataframe(periods=7)

future.tail(7)
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(8)
recovered_forecast_plot = m.plot(forecast)
recovered_forecast_plot = m.plot_components(forecast)