#importing libaries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

import seaborn as sns





from sklearn.model_selection import RandomizedSearchCV , train_test_split

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

plt.style.use('seaborn')

import math

import random

import time

import operator
covid19 = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid19.head()
covid19['ObservationDate']=pd.to_datetime(covid19['ObservationDate'])

covid19['Last Update']=pd.to_datetime(covid19['Last Update'])
covid19.dtypes
grouped = covid19.groupby('ObservationDate')['Last Update', 'Confirmed', 'Deaths'].sum().reset_index()
grouped.head()


fig = px.line(grouped, x="ObservationDate", y="Confirmed", 

              title="Worldwide Confirmed Cases Over Time")

fig.show()



fig = px.line(grouped, x="ObservationDate", y="Confirmed", 

              title="Worldwide Confirmed Cases (Logarithmic Scale) Over Time", 

              log_y=True)

fig.show()
fig = px.line(grouped, x="ObservationDate", y="Deaths", title="Worldwide Deaths Over Time",

             color_discrete_sequence=['#F42272'])

fig.show()



fig = px.line(grouped, x="ObservationDate", y="Deaths", title="Worldwide Deaths (Logarithmic Scale) Over Time", 

              log_y=True, color_discrete_sequence=['#F42272'])

fig.show()
grouped_sl = covid19[covid19['Country/Region'] == "Sri Lanka"].reset_index()

grouped_sl_date = grouped_sl.groupby('ObservationDate')['ObservationDate', 'Confirmed', 'Deaths'].sum().reset_index()
grouped_sl_date.head()
fig = px.line(grouped_sl_date, x="ObservationDate", y="Confirmed", 

              title="Sri Lanka Confirmed Cases Over Time")

fig.show()
grouped_china = covid19[covid19['Country/Region'] == "Mainland China"].reset_index()

grouped_ch_date = grouped_china.groupby('ObservationDate')['ObservationDate', 'Confirmed', 'Deaths'].sum().reset_index()
fig = px.line(grouped_ch_date, x="ObservationDate", y="Confirmed", 

              title="China Confirmed Cases Over Time")

fig.show()
grouped_italy = covid19[covid19['Country/Region'] == "Italy"].reset_index()

grouped_italy_date = grouped_china.groupby('ObservationDate')['ObservationDate', 'Confirmed', 'Deaths'].sum().reset_index()
fig = px.line(grouped_italy_date, x="ObservationDate", y="Confirmed", 

              title="Italy Confirmed Cases Over Time")

fig.show()
covid19['Active'] = covid19['Confirmed'] - (covid19['Deaths'] + covid19['Recovered'])

covid19_new = covid19

without_china = covid19[covid19['Country/Region'] != "Mainland China"]
line_data = covid19_new.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

line_data = line_data.melt(id_vars="ObservationDate", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')

fig = px.line(line_data, x='ObservationDate', y='Count', color='Case', title='Whole World Cases over time')

fig.show()
ch_data = covid19_new[covid19_new['Country/Region'] == 'Mainland China'].reset_index(drop=True)

ch_line_data = ch_data.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
ch_line_data = ch_line_data.melt(id_vars="ObservationDate", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')

fig = px.line(ch_line_data, x='ObservationDate', y='Count', color='Case', title='China Cases over time')

fig.show()
last_data = covid19_new.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

last_data = last_data.reset_index()

last_data = last_data[last_data['ObservationDate'] == max(last_data['ObservationDate'])]

last_data = last_data.reset_index(drop=True)

last_data['Deaths %'] = round(100 * last_data['Deaths'] / last_data['Confirmed'], 2)

last_data['Recovered %'] = round(100 * last_data['Recovered'] / last_data['Confirmed'], 2)

last_data['Active %'] = round(100 * last_data['Active'] / last_data['Confirmed'], 2)

last_data.style.background_gradient(cmap='Pastel1')
pi_data = last_data.melt(id_vars="ObservationDate", value_vars=['Active', 'Deaths', 'Recovered'], var_name='Case', value_name='Count')

fig = px.pie(pi_data, values='Count', names='Case')

fig.show()
wc_data = without_china.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

wc_data = wc_data.reset_index()

wc_data = wc_data[wc_data['ObservationDate'] == max(wc_data['ObservationDate'])]

wc_data = wc_data.reset_index(drop=True)

wc_data['Deaths Rate'] = round(100 * wc_data['Deaths'] / wc_data['Confirmed'], 2)

wc_data['Recovered Rate'] = round(100 * wc_data['Recovered'] / wc_data['Confirmed'], 2)

wc_data['Active Rate'] = round(100 * wc_data['Active'] / wc_data['Confirmed'], 2)

wc_data.style.background_gradient(cmap='Pastel1')
pi_data = wc_data.melt(id_vars="ObservationDate", value_vars=['Active', 'Deaths', 'Recovered'], var_name='Case', value_name='Count')

fig = px.pie(pi_data, values='Count', names='Case')

fig.show()
covid19['Province/State'] = covid19['Province/State'].fillna('')

temp = covid19[[col for col in covid19.columns if col != 'Province/State']]



latest = temp[temp['ObservationDate'] == max(temp['ObservationDate'])].reset_index()

latest_grouped = latest.groupby('Country/Region')['Confirmed', 'Deaths'].sum().reset_index()
fig = px.choropleth(latest_grouped, locations="Country/Region", 

                    locationmode='country names', color="Confirmed", 

                    hover_name="Country/Region", range_color=[1,5000], 

                    color_continuous_scale="peach", 

                    title='Countries with Confirmed Cases')

# fig.update(layout_coloraxis_showscale=False)

fig.show()
wc_bar_data = without_china.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

wc_bar_data = wc_bar_data.drop_duplicates(subset=["Country/Region"], keep='last')



wc_bar_data['Deaths Rate'] = round(100 * wc_bar_data['Deaths'] / wc_bar_data['Confirmed'], 2)

wc_bar_data['Recovered Rate'] = round(100 * wc_bar_data['Recovered'] / wc_bar_data['Confirmed'], 2)

wc_bar_data['Active Rate'] = round(100 * wc_bar_data['Active'] / wc_bar_data['Confirmed'], 2)
wc_bar_data = wc_bar_data[wc_bar_data['Confirmed'] > 1000]

wc_daths_rate = wc_bar_data

wc_daths_rate = wc_daths_rate.sort_values(by=['Deaths Rate'], ascending=False).reset_index(drop=True)

wc_daths_rate.style.background_gradient(cmap='Reds')
cases_per_Day = covid19.groupby(["ObservationDate"])['Confirmed','Deaths', 'Recovered'].sum().reset_index()

sorted_By_Confirmed1=cases_per_Day.sort_values('ObservationDate',ascending=False)



sorted_By_Confirmed1.style.background_gradient(cmap='Reds')
#Grouping different types of cases as per the date

datewise=covid19.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise.head()
import seaborn as sns

sns.set_style("whitegrid")

sns.kdeplot(datewise["Confirmed"])

plt.title("Density Distribution Plot for Confirmed Cases")




sns.kdeplot(datewise["Deaths"])

plt.title("Density Distribution Plot for Death Cases")
sns.kdeplot(datewise["Recovered"])

plt.title("Density Distribution Plot for Death Cases")
#Calculating the Mortality Rate and Recovery Rate

datewise["Mortality Rate"]=(datewise["Deaths"]/datewise["Confirmed"])*100

datewise["Recovery Rate"]=(datewise["Recovered"]/datewise["Confirmed"])*100

 

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,6))

ax1.plot(datewise["Mortality Rate"],label='Mortality Rate')

ax1.axhline(datewise["Mortality Rate"].mean(),linestyle='--',color='black',label="Mean Mortality Rate")

ax1.set_ylabel("Number of Patients")

ax1.set_xlabel("Timestamp")

ax1.legend()



for tick in ax1.get_xticklabels():

    tick.set_rotation(90)

ax2.plot(datewise["Recovery Rate"],label="Recovery Rate")

ax2.axhline(datewise["Recovery Rate"].mean(),linestyle='--',color='black',label="Mean Recovery Rate")

ax2.set_ylabel("Number of Patients")

ax2.set_xlabel("Timestamp")

ax2.legend()

for tick in ax2.get_xticklabels():

    tick.set_rotation(90)
#load dataset

confirm_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv') 

death_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

recoverd_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
cols = confirm_data.keys()

cols
confirmed = confirm_data.loc[:,cols[4]:cols[-1]]

deaths = death_data.loc[:,cols[4]:cols[-1]]

recovered = recoverd_data.loc[:,cols[4]:cols[-1]]
dates = confirmed.keys()

world_cases = []

total_deaths = [] 

mortality_rate = []

recovery_rate = [] 

total_recovered = [] 

total_active = [] 



for i in dates:

    confirmed_sum = confirmed[i].sum()

    death_sum = deaths[i].sum()

    recovered_sum = recovered[i].sum()

    world_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    mortality_rate.append(death_sum/confirmed_sum)

    recovery_rate.append(recovered_sum/confirmed_sum)

    total_recovered.append(recovered_sum)

    total_active.append(confirmed_sum-death_sum-recovered_sum)
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)

total_deaths = np.array(total_deaths).reshape(-1, 1)

total_recovered = np.array(total_recovered).reshape(-1, 1)
#future forcasting for the next 10 days



days_in_future = 10

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)

adjusted_dates = future_forcast[:-10]
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
#split in to train test data

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False) 
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=5, C=0.1)

svm_confirmed.fit(X_train_confirmed, y_train_confirmed)

svm_pred = svm_confirmed.predict(future_forcast)
# check against testing data

svm_test_pred = svm_confirmed.predict(X_test_confirmed)

plt.plot(svm_test_pred)

plt.plot(y_test_confirmed)

print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))

print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
plt.figure(figsize=(12, 6))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast, svm_pred, linestyle='dashed', color='purple')

plt.title('Number of Coronavirus Cases Over Time')

plt.xlabel('Confirm cases vs SVM prediction')

plt.ylabel('Number of Cases')

plt.legend(['Confirmed Cases', 'SVM predictions'])

plt.xticks(size=15)

plt.show()
print('SVM future predictions for 10 Days:')

set(zip(future_forcast_dates[-10:], svm_pred[-10:]))
linear_model = LinearRegression(normalize=True, fit_intercept=False)

linear_model.fit(X_train_confirmed, y_train_confirmed)

test_linear_pred = linear_model.predict(X_test_confirmed)

linear_pred = linear_model.predict(future_forcast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
print(linear_model.coef_)

print(linear_model.intercept_)
plt.plot(test_linear_pred)

plt.plot(y_test_confirmed)
plt.figure(figsize=(12, 6))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast, linear_pred, linestyle='dashed', color='orange')

plt.title('Number of Coronavirus Cases Over Time')

plt.xlabel('Confirm cases vs Linear Regression')

plt.ylabel('Number of Cases')

plt.legend(['Confirmed Cases', 'Linear Regression Predictions'])

plt.xticks(size=15)

plt.show()
# Future predictions using Linear Regression 

print('Linear regression future 10 predictions:')

print(linear_pred[-10:])