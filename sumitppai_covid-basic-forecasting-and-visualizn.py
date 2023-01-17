import numpy as np 

import pandas as pd 

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt 

from sklearn import linear_model
confirmed_cases_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
confirmed_cases_df.columns
deaths_df.columns
confirmed_cases_df.head(10)
date_colums = confirmed_cases_df.columns[4:]
def get_outbreak_details(dataset_type, dataset, window_size=3, outbreak_threshold=50):

    '''Gets the start date of outbreak for each country and aligns the outbreak data from day 0 of outbreak for every country

    

    Parameters:

    -----------

        dataset_type: str

            Type of the dataset (just for printing it)

        dataset: pd.Dataframe

            Dataframe of data to use to predict the outbreak

        window_size: int

            window size (number of preceeding days) for computing rolling mean

        outbreak_threshold: int

            threshold over which we say outbreak has started 

            

    Returns:

    --------

        outbreak_data_df: pd.Dataframe

            Dataframe where each row consists of names of countries where outbreak has started 

            and columns are counts from day 0 (to max 365 days). Last column is the start date of the outbreak.

            

    '''

    # US had county wise data but same was aggregated in State. So we drop county data

    data_df = dataset[~dataset['Province/State'].str.contains(",").fillna(False)]

    # Let's group the data by country (some countries have data by states eg: US, Australia, etc)

    data_grouped_df = data_df.groupby('Country/Region')

    print('Number of countries in the dataset({}):'.format(dataset_type), len(list(data_grouped_df.groups.keys())))

    '''

    print('Countries which have provincial splits:')

    for key, item in data_grouped_df:

        if data_grouped_df.get_group(key).shape[0]>1:

            print('\n \t', key, ':', data_grouped_df.get_group(key)['Province/State'].values, '\n')

    '''

    # aggregate by country and get the total count for each day. Cumulative Counts on Day0, Day1, Day2, ...

    data_counts_total_per_day = data_grouped_df[date_colums].sum()

    # compute the increase per day. Day1-Day0, Day2-Day1, ...

    data_counts_inc_per_day = data_counts_total_per_day.diff(axis=1).drop(['1/22/20'], axis=1)



    # compute the rolling mean with the provided window size

    rolling_mean = data_counts_inc_per_day.rolling(window=window_size, axis=1).mean().fillna(0)

    # get the start date of outbreak - date on which the rolling mean over previous window_size days > outbreak_threshold

    outbreak_countries_with_start = pd.DataFrame(np.where(rolling_mean>=outbreak_threshold)).T.groupby(0)

    number_of_days_max = 365

    outbreak_data_df = pd.DataFrame(columns=np.arange(0, number_of_days_max).tolist() + ['start_date'], 

                                    index=rolling_mean.iloc[list(outbreak_countries_with_start.groups.keys())].index.values)

    for key, item in outbreak_countries_with_start:

        # get the date columns from outbreak start date till current date

        dates = list(rolling_mean.columns[outbreak_countries_with_start.get_group(key)[1].values[0]:])

        # align the data for every country - i.e. start date of outbreak for a country is day 0. 

        outbreak_data_df.loc[rolling_mean.index.values[key]][np.arange(0, number_of_days_max)] = data_counts_total_per_day.loc[rolling_mean.index.values[key]][dates].values.tolist() + [np.nan]*(number_of_days_max-len(dates))

        outbreak_data_df.loc[rolling_mean.index.values[key]]['start_date'] = dates[0]

    print('Number of countries with Outbreaks (rolling mean over {} days >= {}): {}'.format(window_size, 

                                                                                          outbreak_threshold, 

                                                                                          len(outbreak_data_df)))

    

    return outbreak_data_df


outbreak_details_for_confirmed = get_outbreak_details('confirmed', confirmed_cases_df, outbreak_threshold=50)
outbreak_details_for_confirmed.tail(10)
plt.figure(figsize=(15,25))

plt.plot(outbreak_details_for_confirmed.iloc[:, 0:365].T)

plt.legend(outbreak_details_for_confirmed.index)

plt.show()
#plot the counts of confirmed cases in US

plt.plot(np.arange(365), outbreak_details_for_confirmed.loc['US'].values[:-1].T)
#plot log of the counts of the data for US

first_nan_idx = np.where(np.isnan(outbreak_details_for_confirmed.loc['US'].values[:-1].astype(np.float32)))[0][0]

plt.plot(np.arange(first_nan_idx), np.log(outbreak_details_for_confirmed.loc['US'].values[:first_nan_idx].astype(np.float32)))
def perform_plot_regression(dataset, country, past_n=5, forecast_next_m=3):

    ''' Pandemic datasets tends to increase exponentially before slowing down. 

        I will first convert this to log domain and I am assuming that this data will show 

        linear behavior over preceeding few days (around 5). With this assumption we 

        perform simple linear regression

    

    Parameters:

    -----------

        dataset: pd.Dataframe

            Dataframe of data to use to predict the outbreak

        country: str

            Name of the country

        past_n: int

            past n days data on which regression will be performed

        forecast_next_m: int

            next m days over which we will forecast

            

    Returns:

    --------

        Nothing.

        This function only plots the forecasted points and prints the values on the display

    

    '''

    # get the last day until which we have data (since from that day onwards we have NaNs)

    first_nan_idx = np.where(np.isnan(dataset.loc[country].values[:-1].astype(np.float32)))[0][0]

    # we should have data for atleast past n days to do the regression

    if first_nan_idx>=past_n:

        # create the model

        reg = linear_model.LinearRegression()

        # fit on log(counts) - since the curve tends to be exponential

        reg.fit(np.arange(first_nan_idx-past_n, first_nan_idx).reshape(-1, 1), 

                np.log(dataset.loc[country].values[first_nan_idx-past_n:first_nan_idx].astype(np.float32)))



        predicted_m_days = np.round(np.exp(reg.predict(np.arange(first_nan_idx, first_nan_idx+forecast_next_m).reshape(-1, 1))))



        plt.plot(np.arange(first_nan_idx, first_nan_idx + forecast_next_m), predicted_m_days, '.')

        print('Today: {}. Predictions over next {} days for {}:{}'.format( dataset.loc[country].values[first_nan_idx-1],

                                                                          forecast_next_m, 

                                                                          country, 

                                                                          predicted_m_days))

    else:

        print('Today: {} for {}. Not enough data for forecasting.'.format(dataset.loc[country].values[first_nan_idx-1],

                                       country))
for country in outbreak_details_for_confirmed.index.values:

    

    plt.plot(outbreak_details_for_confirmed.loc[['Italy', country], np.arange(365)].T)

    perform_plot_regression(outbreak_details_for_confirmed, country, 5, 3)

    

    plt.legend(['Italy', country])

    plt.show()
fig = plt.figure(figsize=(15,10))

countries_list = ['France', 'Germany', 'Spain', 'United Kingdom', 'US', 'India']

fig.suptitle('(Outbreak) Italy start date:{}'.format(outbreak_details_for_confirmed.loc['Italy']['start_date']))

for i, country in enumerate(countries_list):

    plt.subplot(2,3,i+1)

    plt.plot(outbreak_details_for_confirmed.loc[['Italy', country], np.arange(365)].T)

    perform_plot_regression(outbreak_details_for_confirmed, country, 5, 3)

    plt.legend(['Italy', country])

    

    plt.title('{} start date:{}'.format(country, outbreak_details_for_confirmed.loc[country]['start_date']))



plt.show()
outbreak_details_for_deaths = get_outbreak_details('deaths', deaths_df, outbreak_threshold=25)
outbreak_details_for_deaths
plt.figure(figsize=(15,15))

plt.plot(outbreak_details_for_deaths.iloc[:, 0:365].T)

plt.legend(outbreak_details_for_deaths.index)

plt.title('Outbreak of Deaths')

plt.xlabel('Day n of outbreak')

plt.ylabel('Count')

plt.show()
for country in outbreak_details_for_deaths.index.values:

    plt.plot(outbreak_details_for_deaths.loc[['Italy', country], np.arange(365)].T)

    perform_plot_regression(outbreak_details_for_deaths, country, 5, 3)

        

    plt.legend(['Italy', country])

    plt.xlabel('Day n of outbreak')

    plt.ylabel('Count')

    plt.show()

    
from sklearn import linear_model

import numpy as np





fig = plt.figure(figsize=(15,10))

countries_list = ['US', 'Spain', 'Germany', 'United Kingdom', 'France', 'Netherlands']

fig.suptitle('(Deaths) Italy start date:{}'.format(outbreak_details_for_deaths.loc['Italy']['start_date']))

for i, country in enumerate(countries_list):

    plt.subplot(2,3,i+1)

    plt.plot(outbreak_details_for_deaths.loc[['Italy', country], np.arange(365)].T)

    perform_plot_regression(outbreak_details_for_deaths, country, 5, 3)

    plt.legend(['Italy', country])

    plt.title('{} start date:{}'.format(country, outbreak_details_for_deaths.loc[country]['start_date']))



plt.show()