import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go 

from plotly.offline import init_notebook_mode

import datetime



init_notebook_mode(connected=True)
pd.set_option('display.max_rows', 300)

pd.set_option('display.max_columns', 100)
import warnings

warnings.filterwarnings('ignore')
print(os.listdir('../input/covid19-data-16mar20/'))

os.getcwd()



filename = '../input/covid19-data-16mar20/covid19_data_16mar20.csv'
# os.listdir()

# filename = "covid19_data_16mar20.csv"

covid19 = pd.read_csv(filename, parse_dates=['observation_date'])

print(covid19.shape)
covid19.head(3)
min_date = covid19.observation_date.min(); max_date = covid19.observation_date.max();



print("Data start date: ", min_date)

print("Data end date:   ", max_date)

print("Period of data:  ", max_date - min_date)
fig = px.line(covid19, x="observation_date", y="confirmed", color="country", #line_group="country", 

              hover_name="country",

              line_shape="spline", render_mode="svg",

              title = 'Country wise confirmed cases - "covid19"')

fig.show()
fig = px.line(covid19, x="observation_date", y="deaths", color="country", #line_group="country", 

              hover_name="country",

              line_shape="spline", render_mode="svg",

              title = 'Country wise death cases - "covid19"')

fig.show()
covid19_latest = covid19[covid19.observation_date==max_date]

covid19_latest.shape
covid19_latest.sort_values(['confirmed'], ascending=False, inplace=True)

covid19_latest.reset_index(drop=True, inplace=True)

covid19_latest.head(10)
# Exclude the top infected country - China 

for country in covid19_latest.country[0:10]:

#     print(country)

    # Visualize trend for selected country

    df_plot = covid19[covid19.country==country]

    # print(df_plot)

    fig = go.Figure()



    fig.add_trace(go.Line(x=df_plot["observation_date"], y=df_plot["confirmed"], name="confirmed"))

    fig.add_trace(go.Line(x=df_plot["observation_date"], y=df_plot["deaths"], name="deaths"))



    fig.update_layout(template='none', title={ 'text': 'Confirmed "covid19" cases - [' + country + ']'}

                      , xaxis_title = 'Date', yaxis_title='Counts')



    fig.show()
covid19_latest[covid19_latest.country=='India']
df_plot = covid19[covid19.country=='India']

# print(df_plot)

fig = go.Figure()



fig.add_trace(go.Line(x=df_plot["observation_date"], y=df_plot["confirmed"], name="confirmed"))

fig.add_trace(go.Line(x=df_plot["observation_date"], y=df_plot["deaths"], name="deaths"))



fig.update_layout(template='none', title={ 'text': 'Confirmed "covid19" cases - [India]'}

                  , xaxis_title = 'Date', yaxis_title='Counts')



fig.show()
covid19_latest.head(10)
# process ml ready dataset to predict for n days

# WAP for below tasks & process each country one by one.

#1. select a country

#2. get min_date & max_date for this country

#3. get train dates for min_date to (max_date - n days)

#4. get train labels for (min_date + n days) to max_date

#5. convert train dates to numeric days (number), considering each country will have a different start day (day 1).



#6. Include countries which are infected from more than 30 days.
# How many countries are infected from more than 30 days

country_wise_count = pd.DataFrame(covid19.groupby(['country']).nunique()['observation_date'].reset_index())

country_30_days = country_wise_count[country_wise_count.observation_date >= 30]

country_30_days= country_30_days.sort_values('observation_date', ascending = False).reset_index(drop=True)

print(country_30_days.shape)
# Function to create the training dataset with data from countries affected for more than 30 days.

def fn_create_train_dataset(covid_df, country_name = 'China', prediction_period = 7):

    # Filter for given country

    covid_df = covid_df[covid_df.country== country_name]



    # Get min and max dates

    min_date = covid_df.observation_date.min()

    max_date = covid_df.observation_date.max()

    

    # Add response date to dataset

    covid_df['response_date'] = covid_df.observation_date + datetime.timedelta(days=prediction_period)

    

    # Add numeric date index

    # Since we only have one record for each day, adding an index would do.

    covid_df = covid_df.reset_index(drop=True).reset_index()

    covid_df =covid_df.rename(columns={'index':'date_index'})

    

    # Create response dataset

    covid_df_response = covid_df[['observation_date',

                                  'confirmed']].rename(columns={'observation_date':'response_date',

                                                                'confirmed':'future_cases'})

    

    # Create training dataset

    df_train = pd.merge(covid_df, covid_df_response, on ='response_date')

    

    # drop additional features

    del [df_train['observation_date'], df_train['response_date'], df_train['country']]

    

    return(df_train)
# Test above function

# fn_creat_train_dataset(covid_df = covid19, country_name='China').head(2)
df_train = pd.DataFrame()



for country in country_30_days.country:

    df_temp = fn_create_train_dataset(covid_df = covid19, 

                                     country_name = country)

    df_train = df_train.append(df_temp)

    

print(df_train.shape)

# df_train.head()
df_train_agg_30days = df_train.groupby('date_index').mean()[['confirmed', 'future_cases']].reset_index()



X_30days = df_train_agg_30days[['confirmed']]

y_30days = df_train_agg_30days['future_cases']
# Visualize

df_train_agg_30days.plot(x = 'date_index', y = 'confirmed')

plt.xlabel('Days')

plt.ylabel('Avg Confirmed cases')

plt.title('Aggregated Spread trend - Countries effected for > 30 days')

plt.show()
df_train1 = pd.DataFrame()

for country in covid19['country'].unique():

    df_temp = fn_create_train_dataset(covid_df = covid19, 

                                     country_name = country)

    df_train1 = df_train1.append(df_temp)



# aggregate on all available countries

df_train_agg_all = df_train1.groupby('date_index').mean()[['confirmed', 'future_cases']].reset_index()

# print(df_train_agg_all.shape)

# print(df_train_agg_all.head(2))



X_all = df_train_agg_all[['confirmed']]

y_all = df_train_agg_all['future_cases']
# Visualize

df_train_agg_all.plot(x = 'date_index', y = 'confirmed')

plt.xlabel('Days')

plt.ylabel('Avg Confirmed cases')

plt.title('Aggregated Spread trend - All countries')

plt.show()
country_name = 'Germany'

# Get data for a specific country

df_train_country= fn_create_train_dataset(covid_df = covid19, 

                                     country_name = country_name)



# Train on specific country

X_country = df_train_country[['confirmed']]

y_country = df_train_country['future_cases']
# Visualize

df_train_country.plot(x = 'date_index', y = 'confirmed')

plt.xlabel('Days')

plt.ylabel('Avg Confirmed cases')

plt.title('Country specific spread trend - ' + country_name)

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
########## CHOOSE A DATASET HERE #######################

# Fit polynomial regression to the train_30days dataset

# X = X_30days

# y = y_30days



# Fit polynomial regression to the train_all dataset

# X = X_all

# y = y_all



# Fit polynomial regression to country specific dataset

X = X_country

y = y_country



poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg = LinearRegression()

lin_reg.fit(X_poly, y)





# Visualize

plt.scatter(X, y, color='red')

plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='blue')

plt.title('Current vs predicted')

plt.xlabel('Current Count')

plt.ylabel('Predicted Count')

plt.show()
df_india =covid19[covid19.country == 'India']

# X_india = df_india[['confirmed']]
X_india = df_india.sort_values(['observation_date'], ascending=False).iloc[0:1, 2:3]

lin_reg.predict(poly_reg.fit_transform(X_india))