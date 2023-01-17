import pandas as pd

import numpy as np

from datetime import datetime

import requests

import warnings

from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

import seaborn as sns



from IPython.display import Image

warnings.filterwarnings('ignore')
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-04-2020.csv')
confirmed_pt = confirmed_df[confirmed_df['Country/Region'] == 'Portugal']

deaths_pt = deaths_df[deaths_df['Country/Region'] == 'Portugal']

recovered_pt = recovered_df[recovered_df['Country/Region'] == 'Portugal']

latest_pt = latest_data[latest_data['Country_Region'] == 'Portugal']
def transform_df(df):

    """

    Transforms pivot table to dataframe. Compute

    new columns

    """

    # 2 of March 2020

    dates = df.columns[44:]

    cases = df[dates].iloc[0].values

    new_df = pd.DataFrame({'Date': dates, 'Cases': cases})

    new_df['Date'] = pd.to_datetime(new_df['Date'])

    new_df['Date'] = new_df.Date.apply(lambda x : x.strftime("%d/%m"))

    new_df['DayReportedCases'] = new_df.Cases.diff(periods=1)

    new_df['7dayAvg'] = new_df.Cases.rolling(7).mean().fillna(0).apply(lambda x : round(x, 2)).values

    new_df['PercentageChange'] = new_df['Cases'].pct_change().fillna(0).apply(lambda x : round((x * 100), 0))

    #c = new_df.DayReportedCases.sum()

    #new_df['PerMillionNorm'] = new_df.apply(lambda x : (x['Cases'] * (10 ** 6)) / c ,axis=1)

    new_df.fillna(0, inplace=True)

    return new_df
new_confirmed_pt = transform_df(confirmed_pt)

new_deaths_pt = transform_df(deaths_pt)

new_recovered_pt = transform_df(recovered_pt)
def plot_function(x, y, title, ylabel, xlabel):

    """

    One plot function with pyplot. Receives as arguments x, y and plot info like title 

    ylabel and xlabel

    """

    

    plt.figure(figsize=(25, 4))

    ax = plt.subplot()

    plt.plot(x, y)

    plt.title(title)

    plt.ylabel(ylabel)

    plt.xlabel(xlabel)

    plt.show()
def plot_2_function(x, y, y1, legend1, legend2):

    """

    Two plots in one plot. Receives as arguments x, y and y1. legend1 and legend2 

    refers to info about each plot

    """

    plt.figure(figsize=(30,4))

    ax=plt.subplot()

    plt.plot(x, y)

    plt.plot(x, y1, marker='o')

    plt.legend([legend1, legend2])

    plt.show()
plot_function(new_confirmed_pt.Date, new_confirmed_pt.Cases, 'Evolution of confirmed positive cases everyday', 'Positive Cases', 'Date')
def get_from_data(df, countryname, c):

    df = df[df['Country/Region'] == countryname].groupby('Country/Region').sum()

    df = transform_df(df)

    df['PerMillionNorm'] = df.apply(lambda x : (x['Cases'] * (10 ** 6)) / c ,axis=1)

    return df
new_confirmed_pt = get_from_data(confirmed_df, 'Portugal', 10000000)

new_confirmed_fr = get_from_data(confirmed_df, 'France', 67000000)

#new_confirmed_us = get_from_data(confirmed_df, 'US')

new_confirmed_es = get_from_data(confirmed_df, 'Spain', 47000000)

new_confirmed_uk = get_from_data(confirmed_df, 'United Kingdom', 66000000)

new_confirmed_it = get_from_data(confirmed_df, 'Italy', 60000000)

new_confirmed_fn = get_from_data(confirmed_df, 'Finland', 5500000)

new_confirmed_sw = get_from_data(confirmed_df, 'Sweden', 10000000)

new_confirmed_be = get_from_data(confirmed_df, 'Belgium', 11000000)





new_confirmed_lx = get_from_data(confirmed_df, 'Luxembourg', 600000)

new_confirmed_ger = get_from_data(confirmed_df, 'Germany', 83000000)

new_confirmed_au = get_from_data(confirmed_df, 'Austria', 8000000)

new_confirmed_swi = get_from_data(confirmed_df, 'Switzerland', 8000000)

new_confirmed_neth = get_from_data(confirmed_df, 'Netherlands', 8000000)

new_confirmed_gree = get_from_data(confirmed_df, 'Greece', 10000000)
x = new_confirmed_pt.Date

ypt = new_confirmed_pt['PerMillionNorm']

yfr = new_confirmed_fr['PerMillionNorm']

yes = new_confirmed_es['PerMillionNorm']

#yus = new_confirmed_us['PerMillionNorm']

yuk = new_confirmed_uk['PerMillionNorm']

yit = new_confirmed_it['PerMillionNorm']

yfn = new_confirmed_fn['PerMillionNorm']

ysw = new_confirmed_sw['PerMillionNorm']

ybe = new_confirmed_be['PerMillionNorm']



fig = plt.figure(figsize=(30,15))

ax = plt.subplot()

plt.plot(x, ypt, color='red', marker= 'o', mfc='green', mec='yellow')

plt.plot(x, yfr, color='darkblue')

plt.plot(x, yes, color='orange')

plt.plot(x, yuk, color='darkred')

plt.plot(x, yit, color='green')

plt.plot(x, yfn, color='black')

plt.plot(x, ysw, color='darkorange')

plt.plot(x, ybe, color='black', marker= 'o', mfc='orange', mec='yellow')

plt.legend(['Portugal', 'France', 'Spain', 'UK', 'Italy', 'Finland', 'Sweden', 'Belgium'])

plt.ylabel('Date')

plt.xlabel('Number of positive cases per million people')

plt.show()
x = new_confirmed_pt.Date

ypt = new_confirmed_pt['PerMillionNorm']

ylx = new_confirmed_lx['PerMillionNorm']

yger = new_confirmed_ger['PerMillionNorm']

yau = new_confirmed_au['PerMillionNorm']

yswi = new_confirmed_swi['PerMillionNorm']

yneth = new_confirmed_neth['PerMillionNorm']

ygree = new_confirmed_gree['PerMillionNorm']



fig = plt.figure(figsize=(30,15))

ax = plt.subplot()

plt.plot(x, ypt, color='red', marker= 'o', mfc='green', mec='yellow')

plt.plot(x, ylx, color='darkblue')

plt.plot(x, yger, color='orange')

plt.plot(x, yau, color='darkred')

plt.plot(x, yswi, color='green')

plt.plot(x, yneth, color='black')

plt.plot(x, ygree, color='darkorange')

plt.legend(['Portugal', 'Luxembourg', 'Germany', 'Austria', 'Switzerland', 'Netherlands', 'Greece'])

plt.ylabel('Date')

plt.xlabel('Number of positive cases per million people')

plt.show()
new_deaths_pt = get_from_data(deaths_df, 'Portugal', 10000000)

new_deaths_fr = get_from_data(deaths_df, 'France', 67000000)

#new_confirmed_us = get_from_data(confirmed_df, 'US')

new_deaths_es = get_from_data(deaths_df, 'Spain', 47000000)

new_deaths_uk = get_from_data(deaths_df, 'United Kingdom', 66000000)

new_deaths_it = get_from_data(deaths_df, 'Italy', 60000000)

new_deaths_fn = get_from_data(deaths_df, 'Finland', 5500000)

new_deaths_sw = get_from_data(deaths_df, 'Sweden', 10000000)

new_deaths_be = get_from_data(deaths_df, 'Belgium', 11000000)



new_deaths_lx = get_from_data(deaths_df, 'Luxembourg', 600000)

new_deaths_ger = get_from_data(deaths_df, 'Germany', 83000000)

new_deaths_au = get_from_data(deaths_df, 'Austria', 8000000)

new_deaths_swi = get_from_data(deaths_df, 'Switzerland', 8000000)

new_deaths_neth = get_from_data(deaths_df, 'Netherlands', 8000000)

new_deaths_gree = get_from_data(deaths_df, 'Greece', 10000000)
x = new_deaths_pt.Date

ypt = new_deaths_pt['PerMillionNorm']

yfr = new_deaths_fr['PerMillionNorm']

yes = new_deaths_es['PerMillionNorm']

#yus = new_confirmed_us['PerMillionNorm']

yuk = new_deaths_uk['PerMillionNorm']

yit = new_deaths_it['PerMillionNorm']

yfn = new_deaths_fn['PerMillionNorm']

ysw = new_deaths_sw['PerMillionNorm']

ybe = new_deaths_be['PerMillionNorm']



fig = plt.figure(figsize=(30,15))

ax = plt.subplot()

plt.plot(x, ypt, color='red', marker= 'o', mfc='green', mec='yellow')

plt.plot(x, yfr, color='darkblue')

plt.plot(x, yes, color='orange')

plt.plot(x, yuk, color='darkred')

plt.plot(x, yit, color='green')

plt.plot(x, yfn, color='black')

plt.plot(x, ysw, color='darkorange')

plt.plot(x, ybe, color='black', marker= 'o', mfc='orange', mec='yellow')

plt.legend(['Portugal', 'France', 'Spain', 'UK', 'Italy', 'Finland', 'Sweden', 'Belgium'])

plt.ylabel('Date')

plt.xlabel('Death toll per million people')

plt.show()
x = new_deaths_pt.Date

ypt = new_deaths_pt['PerMillionNorm']

ylx = new_deaths_lx['PerMillionNorm']

yger = new_deaths_ger['PerMillionNorm']

yau = new_deaths_au['PerMillionNorm']

yswi = new_deaths_swi['PerMillionNorm']

yneth = new_deaths_neth['PerMillionNorm']

ygree = new_deaths_gree['PerMillionNorm']



fig = plt.figure(figsize=(30,15))

ax = plt.subplot()

plt.plot(x, ypt, color='red', marker= 'o', mfc='green', mec='yellow')

plt.plot(x, ylx, color='darkblue')

plt.plot(x, yger, color='orange')

plt.plot(x, yau, color='darkred')

plt.plot(x, yswi, color='green')

plt.plot(x, yneth, color='black')

plt.plot(x, ygree, color='darkorange')

plt.legend(['Portugal', 'Luxembourg', 'German', 'Austria', 'Switzerland', 'Netherlands', 'Greece'])

plt.ylabel('Date')

plt.xlabel('Death toll per million people')

plt.show()
x = new_confirmed_pt.Date

y = new_confirmed_pt.Cases

y1 = new_confirmed_pt['7dayAvg']



plt.figure(figsize=(30,4))

ax=plt.subplot()

plt.plot(x, y)

plt.plot(x, y1, marker='o')

plt.legend(['Cases per day', '7 day average cases'])

plt.ylabel('Date')

plt.xlabel('Positive Cases')

plt.show()
plot_function(new_confirmed_pt.Date, new_confirmed_pt['PercentageChange'], 'Percentage increase of positive cases each day', 'Percentage', 'Date')
new_confirmed_pt.tail()
plot_function(new_confirmed_pt.Date, new_confirmed_pt['DayReportedCases'], 'Number of +cases reported each day', 'Positive cases reported', 'Date')
plot_function(new_recovered_pt.Date, new_recovered_pt.Cases, 'Evolution of Recovered cases', 'Freq of recovered', 'Date')
plot_2_function(new_recovered_pt.Date, new_recovered_pt.Cases, new_recovered_pt['7dayAvg'], 'Cases per day', '7 day average cases')
x = new_recovered_pt[new_recovered_pt['PercentageChange'] != 0].Date

y = new_recovered_pt[new_recovered_pt['PercentageChange'] != 0]['PercentageChange']



plt.figure(figsize=(25, 4))

ax = plt.subplot()

plt.bar(x, y)

plt.title('Percentage change each day')

plt.ylabel('Percentage')

plt.xlabel('Dates')

plt.show()
new_confirmed_pt['DaysSinceBeggining'] = new_confirmed_pt.index
X = np.array(new_confirmed_pt.DaysSinceBeggining).reshape(-1, 1)

y = new_confirmed_pt.Cases



simple_lrmodel = LinearRegression()

simple_lrmodel.fit(X, y)

y_fromlr = simple_lrmodel.predict(X)
plt.figure(figsize=(30, 6))

plt.scatter(X, y)

plt.plot(X, y_fromlr, color='red')

plt.legend(['Linear regression line', 'Number of Cases'])

plt.show()
days_max = new_confirmed_pt['DaysSinceBeggining'].max()

X_future = np.array(range(days_max, days_max + 30)).reshape(-1, 1)



y_future = simple_lrmodel.predict(X_future)
plt.figure(figsize=(30, 6))

plt.scatter(X_future, y_future, color='red')

plt.ylabel('Total Positive Cases')

plt.xlabel('Days Since First Positive Case')

plt.show()
plt.matshow(new_confirmed_pt.corr())

plt.show()