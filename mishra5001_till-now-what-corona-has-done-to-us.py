# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import matplotlib.pyplot as plt

import datetime

import seaborn as sns

import plotly.graph_objects as go

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid_19 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covid_19.head()
# We will first try to prepare the data for exploration and then will look to do some predictions



covid_19 = covid_19.drop(['SNo'],axis=1)



# Defining a function to check the NULL Values



def check_null_values(dataframe_to_check):

    null_value  = dataframe_to_check.isnull().sum()

    null_value_precent = round(100*(dataframe_to_check.isnull().sum()/len(dataframe_to_check.index)),2)

    df_to_return  = pd.DataFrame({'Missing Value Count': null_value,'Missing Value Percentage':null_value_precent})

    print(df_to_return)
print(covid_19.shape)



# Checking the Missing Value Distribution

check_null_values(covid_19)
# Now we will look at the Distribution of the State and Provinces to find the Most Affected Places. We can similarly use this technique of Mode to Impute the missing value so that we donot loose that 40% of Data.



covid_19['Province/State'].value_counts()
covid_19['Province/State'] = covid_19['Province/State'].fillna('Location Not Disclosed',axis=0)



check_null_values(covid_19)
# Now just for some Verification we will check if we have the Same Count of Missing Value or Not



print(covid_19[covid_19['Province/State'] == 'Location Not Disclosed'].shape)



# Hence We have 2462 Intact Places, Let's plot Them Now!
fig = px.scatter(covid_19, y="Deaths",x = "Recovered", color="Country/Region",

                 size='Confirmed', hover_data=['Province/State','Confirmed','Deaths','Recovered'],log_x=True,log_y=True)

fig.show()
confirmed_df = covid_19.groupby(['Country/Region','Province/State']).agg({'Confirmed':sum}).reset_index()



confirmed_df.shape
print('Uniqe Countries in our Summed up Data Set:',confirmed_df['Country/Region'].nunique())



print('Unique Countries in our Original Data Set:',covid_19['Country/Region'].nunique())
# Similarly



deaths_df = covid_19.groupby(['Country/Region','Province/State']).agg({'Deaths':sum}).reset_index()

recovered_df = covid_19.groupby(['Country/Region','Province/State']).agg({'Recovered':sum}).reset_index()



print('Uniqe Countries in our Summed up Data Set:',deaths_df['Country/Region'].nunique())



print('Unique Countries in our Original Data Set:',covid_19['Country/Region'].nunique())
# Next we will merge all these to form the correct dataframe



covid_19_1 = confirmed_df.merge(deaths_df,how='inner',on=['Country/Region','Province/State'])

covid_19_df = covid_19_1.merge(recovered_df,how='inner',on=['Country/Region','Province/State'])



check_null_values(covid_19_df)
# Let's have a look at Data



covid_19_df.head(15)
# Now let's plot it again

df_df = covid_19_df.groupby(['Country/Region']).agg({'Deaths':sum,'Confirmed':sum,'Recovered':sum}).reset_index()



fig = px.scatter(df_df, y="Deaths",x = "Recovered", color="Country/Region",

                 size='Confirmed', hover_data=['Confirmed','Deaths','Recovered'])

fig.update_yaxes(nticks=20)

fig.update_xaxes(nticks=50)



fig.show()
covid_19_time_analysis = covid_19.loc[covid_19['Country/Region'].isin(['Mainland China','Italy','Iran','South Korea','Spain'])]



covid_19_time_analysis.shape
print(covid_19_time_analysis.info())



# We will drop Province as we donot need that column, since we will be having the Country in common for those Provinces



covid_19_time_analysis = covid_19_time_analysis.drop(['Province/State'],axis=1)

covid_19_time_analysis.rename({'Country/Region':'Country'},inplace=True,axis=1)

covid_19_time_analysis.head()
covid_19_time_analysis['Month of Observation'] = pd.to_datetime(covid_19_time_analysis['ObservationDate']).dt.strftime('%B')

#covid_19_time_analysis['Month of Observation'] = covid_19_time_analysis['Month of Observation'].dt.strftime('%B')

covid_19_time_analysis['Year of Observation'] = pd.to_datetime(covid_19_time_analysis['ObservationDate']).dt.year
print(covid_19_time_analysis.describe())



# We will also look for the Unique values in the created columns



print('Unique Values in Month of Observation',covid_19_time_analysis['Month of Observation'].nunique())

print('Unique Values in Month of Observation',covid_19_time_analysis['Year of Observation'].nunique())



# This means that we have data from January to March and for the Year of 2020 only.
covid_19_time_analysis['Hour of Observation'] = pd.to_datetime(covid_19_time_analysis['Last Update']).dt.hour
covid_19_time_analysis.head(10)
fig = px.bar(covid_19_time_analysis.groupby('Country').get_group('Mainland China'), x='Hour of Observation', y='Confirmed', color='Month of Observation')

fig.update_xaxes(nticks=24)

fig.show()
fig = px.bar(covid_19_time_analysis, x="Hour of Observation", y="Confirmed", facet_col="Country",color='Month of Observation',log_y=True)

fig.update_xaxes(nticks=6)

#fig.update_yaxes(nticks=20)

fig.show()
fig = px.bar(covid_19_time_analysis, x="Hour of Observation", y="Deaths", facet_col="Country",color='Month of Observation',log_y=True,hover_data=[

    'Country','Confirmed','Hour of Observation','Month of Observation'

])

fig.update_xaxes(nticks=6)

#fig.update_yaxes(nticks=50)

fig.show()
fig = px.bar(covid_19_time_analysis, x="Hour of Observation", y="Recovered", facet_col="Country",color='Month of Observation',log_y=True,hover_data=[

    'Country','Confirmed','Hour of Observation','Month of Observation','Deaths','Recovered'],title='Looking at the Countries based on Recovered Rate and Hourly Cases for different Months!'

)

fig.update_xaxes(nticks=6)

#fig.update_yaxes(nticks=50)

fig.show()
fig = px.scatter(covid_19_time_analysis, x='ObservationDate', y='Deaths', color='Country',title='Looking at the Countries based on Observation Rate and Death Cases!')

fig.update_yaxes(nticks=20)

fig.show()
fig = px.scatter(covid_19_time_analysis, x='ObservationDate', y='Recovered', color='Country',title='Looking at the Countries based on Observation Rate and Recovered Cases!')

fig.update_yaxes(nticks=20)

fig.show()
fig = px.scatter(covid_19_time_analysis, x='ObservationDate', y='Confirmed', color='Country',title='Looking at the Countries based on Observation Rate and Confirmed Cases!')

fig.update_yaxes(nticks=20)

fig.show()
covid_19_time_analysis.head(10)
city_wise = covid_19_time_analysis.groupby('Country').sum()

city_wise['Death Rate'] = city_wise['Deaths'] / city_wise['Confirmed'] * 100

city_wise['Recovery Rate'] = city_wise['Recovered'] / city_wise['Confirmed'] * 100

city_wise['Active'] = city_wise['Confirmed'] - city_wise['Deaths'] - city_wise['Recovered']

city_wise = city_wise.sort_values('Deaths', ascending=False).reset_index()

city_wise
px.scatter(city_wise,y = 'Recovery Rate',color='Country',x='Active',size='Confirmed',title='Looking at the Countries based on Recovery Rate and Active Cases!')

fig = go.Figure()

fig.add_trace(go.Scatter(x=city_wise['Recovery Rate'], y=city_wise['Country'],

                    mode='lines+markers',

                    name='Recovery rate'))

fig.add_trace(go.Scatter(x=city_wise['Death Rate'], y=city_wise['Country'],

                    mode='lines+markers',

                    name='Death Rate'))

fig.show()
# Now we will try to Look for the 

from plotly.subplots import make_subplots



confirm_death_recovery_cases = covid_19_time_analysis.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].sum().reset_index()



plot = make_subplots(rows=1, cols=3, subplot_titles=("Comfirmed", "Deaths", "Recovered"))



subPlot1 = go.Scatter(

                x=confirm_death_recovery_cases['ObservationDate'],

                y=confirm_death_recovery_cases['Confirmed'],

                name="Confirmed",

                line=dict(color='royalblue', width=4, dash='dot'),

                opacity=0.8)



subPlot2 = go.Scatter(

                x=confirm_death_recovery_cases['ObservationDate'],

                y=confirm_death_recovery_cases['Deaths'],

                name="Deaths",

                line=dict(color='red', width=4, dash='dot'),

                opacity=0.8)



subPlot3 = go.Scatter(

                x=confirm_death_recovery_cases['ObservationDate'],

                y=confirm_death_recovery_cases['Recovered'],

                name="Recovered",

                line=dict(color='firebrick', width=4, dash='dash'),

                opacity=0.8)



plot.append_trace(subPlot1, 1, 1)

plot.append_trace(subPlot2, 1, 2)

plot.append_trace(subPlot3, 1, 3)

plot.update_layout(template="ggplot2", title_text = '<b><i>Global Spread of the Covid 19 Over Time</i></b>',xaxis_title='Observation Dates',

                   yaxis_title='Count')



plot.show()
fig = px.pie(city_wise, values='Recovery Rate', names='Country')

fig.show()
fig = make_subplots(rows=1, cols=2,specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=city_wise['Country'], values=city_wise['Death Rate'], name="Death Rate"),

              1, 1)

fig.add_trace(go.Pie(labels=city_wise['Country'], values=city_wise['Recovery Rate'], name="Recovery Rate"),

              1, 2)

fig.update_traces(hole=.4, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Death and Recovery rate for Different Countries!",

    annotations=[dict(text='Death Rate', x=0.18, y=0.5, font_size=12, showarrow=False),

                 dict(text='Recovery Rate', x=0.82, y=0.5, font_size=12, showarrow=False)])

fig.show()
recovered_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

confirmed_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

deaths_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
confirmed_df.head()
cols = confirmed_df.keys()

print(cols)

# gettingg all the Dates



confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]

deaths = deaths_df.loc[:, cols[4]:cols[-1]]

recoveries = recovered_df.loc[:, cols[4]:cols[-1]]
over_time_date = confirmed.keys()



# Number of cases Over time

world_cases = []

# Number of deaths Over time

total_deaths = [] 

# The Rate at which Death id occuring

mortality_rate = []

# Number of people Recovered Over Time

total_recovered = [] 



for i in over_time_date:

    confirmed_sum = confirmed[i].sum()

    death_sum = deaths[i].sum()

    recovered_sum = recoveries[i].sum()

    world_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    mortality_rate.append(death_sum/confirmed_sum)

    total_recovered.append(recovered_sum)



# Converting them into Array for Modelling



from_starting = np.array([i for i in range(len(over_time_date))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)
moving_day_span = 7

single_week_prediction = np.array([i for i in range(len(over_time_date)+moving_day_span)]).reshape(-1, 1)

modified_date = single_week_prediction[:-7]
start_date_data = '1/22/2020'

start_date = datetime.datetime.strptime(start_date_data, '%m/%d/%Y')

in_future_dates = []

for i in range(len(single_week_prediction)):

    in_future_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))



X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(from_starting, world_cases, test_size=0.15, shuffle=False)



print(X_train_confirmed.shape)

print(X_test_confirmed.shape)

print(y_train_confirmed.shape)

print(y_test_confirmed.shape)
kernel = ['poly', 'sigmoid', 'rbf']

c = [0.001, 0.01, 0.1, 1,10]

gamma =[0.001, 0.01, 0.1, 1,10]

epsilon = [0.01, 0.1, 1,10]

shrinking = [True, False]

svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}



svm = SVR()

best_model = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

best_model.fit(X_train_confirmed, y_train_confirmed)
print(best_model.best_params_)



# Now comes, what we have been waiting for, make those predictions



best_estimator_from_model = best_model.best_estimator_

make_predictions = best_estimator_from_model.predict(single_week_prediction)
plt.figure(figsize=(20, 8))

plt.plot(modified_date, world_cases)

plt.plot(single_week_prediction, make_predictions, linestyle='dashed', color='red')

plt.title('# of Coronavirus Cases Over Time', size=15)

plt.xlabel('Days Since 1/22/2020', size=10)

plt.ylabel('Count of Cases', size=10)

plt.legend(['Confirmed Cases', 'Model Predictions'])

plt.xticks(size = 20)

plt.show()
# Now you people can see the precition till 22nd March 2020. Similarly, we would be doing some prediction for Deaths and recoveries



over_time_deaths = deaths.keys()

from_starting_deaths = np.array([i for i in range(len(over_time_deaths))]).reshape(-1, 1)

death_cases = np.array(total_deaths).reshape(-1, 1)



moving_day_span = 7

single_week_prediction = np.array([i for i in range(len(over_time_deaths)+moving_day_span)]).reshape(-1, 1)

modified_date = single_week_prediction[:-7]



start_date_data = '1/22/2020'

start_date = datetime.datetime.strptime(start_date_data, '%m/%d/%Y')

in_future_dates = []

for i in range(len(single_week_prediction)):

    in_future_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))



X_train_confirmed_deaths, X_test_confirmed_deaths, y_train_confirmed_deaths, y_test_confirmed_deaths = train_test_split(from_starting_deaths, death_cases, test_size=0.15, shuffle=False)



print(X_train_confirmed_deaths.shape)

print(X_test_confirmed_deaths.shape)

print(y_train_confirmed_deaths.shape)

print(y_test_confirmed_deaths.shape)
kernel = ['poly', 'sigmoid', 'rbf']

c = [0.001, 0.01, 0.1, 1,10]

gamma =[0.001, 0.01, 0.1, 1,10]

epsilon = [0.01, 0.1, 1,10]

shrinking = [True, False]

svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}



svm = SVR()

best_model = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

best_model.fit(X_train_confirmed_deaths, y_train_confirmed_deaths)
print(best_model.best_params_)



# Now comes, what we have been waiting for, make those predictions



best_estimator_from_model = best_model.best_estimator_

make_predictions = best_estimator_from_model.predict(single_week_prediction)
plt.figure(figsize=(20, 8))

plt.plot(modified_date, death_cases)

plt.plot(single_week_prediction, make_predictions, linestyle='dashed', color='red')

plt.title('# of Coronavirus Death Cases Over Time', size=15)

plt.xlabel('Days Since 1/22/2020', size=10)

plt.ylabel('Count of Death Cases', size=10)

plt.legend(['Death Cases', 'Model Predictions'])

plt.xticks(rotation = 90)

plt.show()
# Now you people can see the precition till 22nd March 2020. Similarly, we would be doing some prediction for Deaths and recoveries



over_time_recoveries = recoveries.keys()

from_starting_recoveries = np.array([i for i in range(len(over_time_recoveries))]).reshape(-1, 1)

recovered_cases = np.array(total_recovered).reshape(-1, 1)



moving_day_span = 7

single_week_prediction = np.array([i for i in range(len(over_time_recoveries)+moving_day_span)]).reshape(-1, 1)

modified_date = single_week_prediction[:-7]



start_date_data = '1/22/2020'

start_date = datetime.datetime.strptime(start_date_data, '%m/%d/%Y')

in_future_dates = []

for i in range(len(single_week_prediction)):

    in_future_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))



X_train_confirmed_recoveries, X_test_confirmed_recoveries, y_train_confirmed_recoveries, y_test_confirmed_recoveries = train_test_split(from_starting_recoveries, recovered_cases, test_size=0.15, shuffle=False)



print(X_train_confirmed_recoveries.shape)

print(X_test_confirmed_recoveries.shape)

print(y_train_confirmed_recoveries.shape)

print(y_test_confirmed_recoveries.shape)
kernel = ['poly', 'sigmoid', 'rbf']

c = [0.001, 0.01, 0.1, 1,10]

gamma =[0.001, 0.01, 0.1, 1,10]

epsilon = [0.01, 0.1, 1,10]

shrinking = [True, False]

svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}



svm = SVR()

best_model = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

best_model.fit(X_train_confirmed_recoveries, y_train_confirmed_recoveries)
print(best_model.best_params_)



# Now comes, what we have been waiting for, make those predictions



best_estimator_from_model = best_model.best_estimator_

make_predictions = best_estimator_from_model.predict(single_week_prediction)
plt.figure(figsize=(20, 8))

plt.plot(modified_date, recovered_cases)

plt.plot(single_week_prediction, make_predictions, linestyle='dashed', color='red')

plt.title('# of Coronavirus Recovered Cases Over Time', size=15)

plt.xlabel('Days Since 1/22/2020', size=10)

plt.ylabel('Count of Recovery Cases', size=10)

plt.legend(['Recovered Cases', 'Model Predictions'])

plt.xticks(rotation = 90)

plt.show()
covid_india = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

covid_india.head()