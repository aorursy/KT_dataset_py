import datetime



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVR
allowed_countries = ['China', 'Italy', 'US', 'Spain', 'France', 'Germany', 'Ukraine', 'Canada']
UNNORMAL_STATISTIC_DIFF_COEF = 2
confirmed_df = pd.read_csv(

    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv(

    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recoveries_df = pd.read_csv(

    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
confirmed_df.head()
deaths_df.head()
recoveries_df.head()
def get_cases_and_dates_for_plot(df: pd.DataFrame) -> (dict, list):

    cols = df.keys()

    data = df.loc[:, cols[4]:cols[-1]]

    dates = data.keys()



    cases = {country: [] for country in df['Country/Region'].unique() if country in allowed_countries}

    world_cases = []

    for i in dates:

        cases_sum = data[i].sum()



        world_cases.append(cases_sum)



        for country in cases.keys():

            cases[country].append(df[df['Country/Region'] == country][i].sum())



    cases['World'] = world_cases



    adjusted_dates = np.array([i for i in range(len(dates))]).reshape(-1, 1)



    return cases, adjusted_dates
def get_unnormal_from_avg_cases_and_avg_increase(cases: list) -> (list, float):

    daily_increases = [(b * 100 / a) - 100 for a, b in zip(cases, cases[1:])]

    avg_daily_increases = round(np.mean(daily_increases), 2)

    return [None] + [cases[i] if el > avg_daily_increases * UNNORMAL_STATISTIC_DIFF_COEF else None for i, el in

                     enumerate(daily_increases)], avg_daily_increases
def show_cases_plot(cases: dict, dates: list, title: str) -> None:

    plt.figure(figsize=(16, 9))

    for value in cases.values():

        plt.plot(dates, value)

    unnormal_from_avg_cases, avg_daily_increase = get_unnormal_from_avg_cases_and_avg_increase(cases['World'])

    plt.plot(dates, unnormal_from_avg_cases, linestyle='dashed', color='black')

    plt.figtext(.15, .4, f"AVG daily increase = {avg_daily_increase}%")

    plt.figtext(.15, .35, f"Cases where increase more than AVG in {UNNORMAL_STATISTIC_DIFF_COEF} times marked dotted")



    plt.title(title, size=20)

    plt.xlabel(f'Days Since {dates[0]}', size=20)

    plt.ylabel('# of Cases', size=30)

    plt.legend(cases.keys(), prop={'size': 20})

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()
confirmed_cases, dates = get_cases_and_dates_for_plot(confirmed_df)

world_cases = np.array(confirmed_cases['World']).reshape(-1, 1)

show_cases_plot(confirmed_cases, dates, 'Number of Coronavirus cases')
show_cases_plot(*get_cases_and_dates_for_plot(deaths_df), 'Number of Coronavirus deaths')
show_cases_plot(*get_cases_and_dates_for_plot(recoveries_df), 'Number of Coronavirus recoveries')
days_in_future = 10

future_forcast = np.array([i for i in range(len(dates) + days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-days_in_future]

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22,

                                                                                            world_cases, test_size=0.05,

                                                                                            shuffle=False)

svm_confirmed = SVR(shrinking=True, kernel='poly', gamma=0.01, epsilon=1, degree=8, C=0.1)

svm_confirmed.fit(X_train_confirmed, y_train_confirmed)

svm_pred = svm_confirmed.predict(future_forcast)
plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast, svm_pred, linestyle='dashed', color='purple')

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['Confirmed Cases', 'SVM predictions'], prop={'size': 20})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()