import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import random

import sys

import datetime



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression



import warnings

if not sys.warnoptions:

    warnings.simplefilter("ignore")
# A function to read and clean the dataser

def read_csv(url, since_cases_num = 10):

    '''

    Read csv file from the COVID-19 Data Repository by Kaggle. 

    Cleans and adds two columns :

        - lgp_cases = np.log1p(df.ConfirmedCases,

        - lgp_deaths = np.log1p(df.Fatalities)

    

    Return two datasets with Date as an index and a sorted list of countries by number of casualities.

    '''

    # read data

    df = pd.read_csv(url)

    

    # Rename culomns 'Country/Region' & the South Korea's name 

    df.rename(columns={'Country_Region' : 'Country'}, inplace=True)

    df['Country'] = df['Country'].replace({'Korea, South': 'South Korea'})

    

    # Grouping by country, rename columns & parse dates

  

    df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')

    df = df.assign(days = df['Date'].dt.dayofyear - 21)

    

    # Generate the second df with the minimum case

    dff = df.groupby(['Country', 'Date']).sum()

    dff = dff.reset_index()

    

    LAST_DATE = dff.iloc[-1, 1]

    

    countries = dff[dff['Date'].eq(LAST_DATE) & dff['Fatalities'].ge(since_cases_num)

           ].sort_values(by='Fatalities', ascending=False)

    countries = countries['Country'].values

    

    COL_X = f'Days since {since_cases_num}th death'

    df_since = dff[dff['Country'].isin(countries)].copy()

    days_since = (df_since.assign(F=df_since['Fatalities'].ge(since_cases_num))

                  .set_index('Date')

                  .groupby('Country')['F'].transform('idxmax'))

    

    df_since[COL_X] = (df_since['Date'] - days_since.values).dt.days.values + 1

    df_since = df_since[df_since[COL_X].ge(0)]

    

    return df.set_index('Date'), df_since.set_index('Date'), list(countries[:7])
url = '/kaggle/input/covid19-global-forecasting-week-4/'

# url_test = '/kaggle/input/covid19-global-forecasting-week-4/test.csv'

# Train dataset

since_cases_num = 10

train, train_since, top_countries = read_csv(url+'train.csv', since_cases_num)

# Test dataset

test = pd.read_csv(url+'test.csv', parse_dates=['Date'], index_col=['Date'])

test = test.rename(columns={'Country_Region' : 'Country'})

test['Country'] = test['Country'].replace({'Korea, South': 'South Korea'})



xsubmission = pd.read_csv(url+'submission.csv')
train.head()
train_since.head()
test.head(3)
def province(df):

    '''

    Replacing empty rows in column 'Province_State' with the corresponding 'Country' and adding 'Country' to the corresponding states if the row is not empty.

    return the cleaned dataset

    '''

    df = df.reset_index()

    df['Province_State'] = df['Province_State'].map(str)

    for row, state in enumerate(df['Province_State']):

        if state == "nan":

            df.loc[row, 'Province_State'] = df.loc[row, 'Country']

        else:

            df.loc[row, 'Province_State'] = df.loc[row, 'Country'] + '_' + df.loc[row, 'Province_State']

    df = df.set_index('Date')

    return df
train = province(train)

test = province(test)

print(f'Number of states :\nTrain = {len(train.Province_State.unique())}\nTest = {len(test.Province_State.unique())}')
train.sample(5)
full_countries = list(train.Country.unique())

full_sc = list(train.Province_State.unique())

num_country = len(full_countries)

num_sc = len(full_sc)



print(f'Number of countries : {num_country}')

print(f'Number of state_countries : {num_sc}')
# List of countries to look at & corresponding colors

colors = [[0,0,0], [255/255,165/255, 0], [86/255,180/255,233/255], [0, 191/255, 255/255],

          [213/255,94/255,0], [0,114/255,178/255], [0,0,128/255]]



# Plotting

plt.style.use('fivethirtyeight')

plt.figure(figsize=(16, 7))

sns.lineplot(x=f'Days since {since_cases_num}th death', y='ConfirmedCases', hue='Country', data=train_since.loc[train_since['Country'].isin(top_countries)], palette=colors)

plt.title('Number of confirmed cases for the selected countries')

plt.show() 
# Plotti,g the number of deaths for the selected countries

plt.figure(figsize=(16, 7))

sns.lineplot(x=f'Days since {since_cases_num}th death', y='Fatalities', hue='Country', data=train_since[train_since['Country'].isin(top_countries)], palette=colors)

plt.title("Number of deaths for the selelcted countries")

plt.xlabel(f'Days after {since_cases_num}th death officially reported')

plt.ylabel('Number of cumulative death')

# plt.ylim(0, 500)

plt.show()
# Creating a list of datasets for the selected countries

selected_df = [train.loc[train['Country'] == country] for country in top_countries]
def df_agg(df, agg=True):

    '''

    Aggregating the dataset by country if agg. And adding two columns:

        - lgp_cases = np.log1p(df.ConfirmedCases),

        - lgp_deaths = np.log1p(df.Fatalities)

    '''

    if agg:

        df = df.reset_index()

        df = df.groupby(['Country', 'Date']).sum()

        df = df.reset_index()

    df = df.assign(lgp_cases = np.log1p(df.ConfirmedCases),

                   lgp_deaths = np.log1p(df.Fatalities))

    return df
def plot_log_CaseDeath(df = selected_df[0], country=top_countries[0], delta=0):

    '''

    Display one of the two following plots:

        - Number of confirmed cases vs. deaths for the corresponding country (log values),

        - Number of confirmed cases vs. deaths for the corresponding country (Ajusted values with a lag of time in days)

    '''

    df = df_agg(df)

    x_1 = np.array(range(df.shape[0]))

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    ax.plot(df.lgp_cases.values)

    ax.plot(df.loc[delta:, 'lgp_deaths'].values)

    if delta == 0:

        plt.title(f'Number of confirmed cases vs. deaths for {country} (log values)')

        plt.legend(('log_confirmed_cases', 'log_deaths'))

    else:

        plt.title(f'Number of confirmed cases vs. deaths for {country}\n(Ajusted values with a lag of {delta} days)')

        plt.legend(('log_confirmed_cases', 'Ajusted_log_deaths'))

    plt.xlabel('Days since the 1st reported case')

    plt.ylabel('Values (log)')

    plt.show()
for idx in range(len(selected_df)):

    plot_log_CaseDeath(df= selected_df[idx], country=top_countries[idx], delta=0)
def get_delta(df=selected_df[0], agg=True):

    '''

    Calculate the lag time in days between the Confirmed Cases and the Fatalities (in log values)

    '''

    if agg:

        df = df_agg(df, agg=True)

    else:

        df = df_agg(df, agg=False)

    for num in range(1, 50):

        shift = pd.DataFrame(data=df.lgp_cases.values - df.lgp_deaths.shift(-num).values, columns=['value'])

        shift.dropna(axis=0, inplace=True)

        if len(shift.query('value >= 0')) != len(shift):   # We need to check if all the values of log_deaths are less than log_cases

            break

        else : 

            sum_shift = shift.sum() / len(shift)

    return num
dlt = []

for df in selected_df:

    dlt.append(get_delta(df))

dlt
for idx in range(len(selected_df)):

    plot_log_CaseDeath(df= selected_df[idx], country=top_countries[idx], delta=dlt[idx])
def linear_mod(df=selected_df[0], country=top_countries[0], delta=dlt[0]):

    '''

    Calculate the linear regression between x = log1p(Confirmed Cases) & y = log1p(deaths).

    Returns df, x, y, slope, intercept.

    '''

    df = df_agg(df)

    df = df.query('lgp_deaths > 0')

    

    y=df.lgp_deaths.iloc[delta:].values

    x=df.lgp_cases.iloc[:y.shape[0]].values

    

    # Find the slope and intercept of the best fit line

    slope, intercept = np.polyfit(x, y, 1)



    # Create a list of values in the best fit line

    abline_values =  [slope * i + intercept for i in x]

        

    # Add the projections to the orginal df 

    d = len(df) - len(x)

    df = df.assign(predictions = 0)

    df.predictions.iloc[d:] = np.expm1(abline_values).astype(int)

    

    df.predictions = pd.to_numeric(df.predictions, errors='coerce')

    df = df.dropna(subset = ['predictions'])    

    

    return df, x, y, slope, intercept, abline_values
X = ['x_' + country for country in top_countries]

Y = ['y_'  + country for country in top_countries]

Slope = ['slope_' + country for country in top_countries]

Intercept = ['intercept_' + country for country in top_countries]

Ab_line = ['abline_' + country for country in top_countries]

predictions = ['predict_' + country for country in top_countries]





for idx in range(len(selected_df)):    

    selected_df[idx], X[idx], Y[idx], Slope[idx], Intercept[idx], Ab_line[idx] = linear_mod(df=selected_df[idx], country=top_countries[idx], delta=dlt[idx])

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    sns.scatterplot(X[idx], Y[idx], ax = ax, )

    sns.lineplot(X[idx], Ab_line[idx], ax = ax)

    plt.title(f'{top_countries[idx]} : The linear regression between log_cases vs. ajusted log_deaths\n {round(Slope[idx], 2)} * x + {round(Intercept[idx], 2)}')

    plt.xlabel('Confirmed Cases (log)')

    plt.ylabel('Reported deaths (log)')

    plt.show()
selected_df[0].head(2)
# Function to plot the predictions vs. observed values

def plot_predict(df=selected_df[0], country=top_countries[0]):

    df = df_agg(df)

    mae = round(mean_absolute_error(df.Fatalities.values, df.predictions.values), 4)

    rmse = round(r2_score(df.Fatalities.values, df.predictions.values), 4)

    x_1 = np.array(range(df.shape[0]))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.plot(df.Fatalities.values)

    ax.plot(df.predictions.values, 'o-')

    plt.title(f'Observed vs. Projected values of the number of deaths in {country}\n mae = {mae} ; rmse = {rmse}')

    plt.legend(('Observed_deaths', 'Predicted_deaths'))

    plt.xlabel('Days since the 1st reported case')

    plt.ylabel('Number of deaths')

    plt.show()
for idx in range(len(selected_df)):

    plot_predict(df=selected_df[idx], country=top_countries[idx])
df = train[['days', 'Province_State', 'ConfirmedCases', 'Fatalities']].reset_index(drop=True).set_index('days')

df.tail(2)
from scipy import interpolate



cases_int = {}

for state in full_sc: 

    data = df.loc[df['Province_State'] == state]

    x = np.arange(0, len(data))

    x_val = list(range(len(x), len(x)+43))

    y = data['ConfirmedCases'].values

    poly = np.polyfit(x, y, deg=5)

    y_hat = np.maximum(0, np.polyval(poly, x_val))

    cases_int[state] = (test.loc[test['Province_State'] == state, 'ForecastId'].values, y_hat.astype(int))
# Create a dictionnary of deltas by state

Delta = {}

for state in full_sc:

    Delta[state] = get_delta(df.loc[df['Province_State'] == state], agg=False)
def model_fit(train, state, label='lgp_deaths', features='lgp_cases', print_=False):

    '''

    Building and training a model for each State/Country.

    Return the corresponding predictions

    '''

    

    delta = Delta[state] 

    df1 = train.loc[train['Province_State'] == state]

    df1 = df_agg(df1, agg=False)



    X = df1.loc[:, features].values

    y = df1.loc[:, label].values

    y_train = y[delta:].reshape(-1, 1)

    X_train = X[:y_train.shape[0]].reshape(-1, 1)

    

    # X_test

    X_test_full =  np.concatenate((X[y_train.shape[0]:], np.log1p(cases_int[state][1])), axis=0)

    X_test = X_test_full[:43].reshape(-1, 1)

    if print_:

        print(f'Delta : {delta} / X shape : {X.shape} / y shape : {y.shape}')

        print(f'X_train shape : {X_train.shape}, y_train shape : {y_train.shape}')

        print(f'X_test_full shape : {X_test_full.shape} / X_test shape : {X_test.shape}')

    

    # Create the model

    model = LinearRegression()

    # Fit the model

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    exp_predictions = np.expm1(predictions)



    return exp_predictions.astype(int)
# Checking the function in the case of Algeria

y_preds = model_fit(train, 'Algeria', label='lgp_deaths', features='lgp_cases', print_=True)



start = np.datetime64('2020-04-09')

x_preds = np.arange(start, start + np.timedelta64(43,'D'))



plt.figure(figsize=(10, 5))

plt.plot(x_preds, y_preds, '--')

plt.title('Forecasted Fatalities for Algeria')

plt.show()
# Looping through the full list of States/Country to forecast number of fatalities

deaths_preds = {}

for state in full_sc:

    y_hat = model_fit(train, state, label='lgp_deaths', features='lgp_cases')

    deaths_preds[state] = (test.loc[test['Province_State'] == state, 'ForecastId'].values, y_hat.ravel())
# Puting it all together

def create_sub():

    x_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

    for state in full_sc:

        df_tmp = pd.DataFrame({'ForecastId': cases_int[state][0], 'ConfirmedCases': cases_int[state][1], 'Fatalities': deaths_preds[state][1]})

        x_out = pd.concat([x_out, df_tmp], axis = 0)

    return x_out
# Submitting

x_out = create_sub()

x_out.ForecastId = x_out.ForecastId.astype('int')

x_out.tail()

x_out.to_csv('submission.csv', index=False)