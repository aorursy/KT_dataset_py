import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display, Markdown

from datetime import datetime, timedelta

from scipy.optimize import curve_fit



def show(txt):

    display(Markdown(txt))



def days(d):

    d1 = datetime.strptime('1/22/20', "%m/%d/%y")

    d2 = datetime.strptime(d, "%m/%d/%y")

    return abs((d2-d1).days)



def day_to_date_str(d):

    d1 = datetime.strptime('1/22/20', "%m/%d/%y")

    return datetime.strftime(d1+timedelta(days=d), "%b %d, %Y")



def logistic(x, L, k, x0):

    return L/(1+np.exp(-k*(x-x0)))



def r_squared(y_true, y_pred):

    mean_y = np.mean(y_true)

    ss_res = np.sum((y_true-y_pred)**2)

    ss_tot = np.sum((y_true-mean_y)**2)

    return 1-ss_res/ss_tot



def analyze_y(covid, yName):

    show(f'### {yName}  \n')

    scale = covid[yName].values[-1]

    if scale == 0:

        show('**Not enough data**')

        return

    x = np.linspace(0, covid.shape[0]-1, num=covid.shape[0])

    y = covid[yName].values/scale

    params, cov = curve_fit(logistic, x, y, maxfev=10000000)

    L, k, x0 = params

    show(f'params|L|k|$x_0$\n---|---|---|---\n|{round(L*scale, 3)}|{round(k, 3)}|{round(x0, 3)}')

    y_pred = logistic(x, *params)

    r_sq = round(r_squared(y, y_pred), 3)

    show(f'$R^2 = {r_sq}$')

    first_day = 0

    for i in range(1, len(x)):

        if y[i] == 0.0:

            first_day = i

        else:

            break

    threshold = 1.000001

    last_day = int(round((-1/k)*np.log((1-threshold)/(threshold*np.exp(-k)-1))+x0))

    plot_x = np.linspace(first_day, last_day, last_day-first_day+1)

    plot_y = logistic(plot_x, *params)*scale

    plt.plot(plot_x, plot_y, 'r-')

    plt.scatter(x[first_day:], y[first_day:]*scale)

    plt.show()

    end_date = day_to_date_str(last_day)

    total = int(round(L*scale))

    show(f'Expected end date|Expected {yName.lower()} by end date\n---|---\n{end_date}|{total}')

    std_dev = np.sqrt(np.mean((scale*(y_pred-y))**2))

    show(f'Expected {yName.lower()} in next 3 days (std dev = {int(round(std_dev))}):')

    today = scale

    day1 = int(round(logistic(x[-1]+1, *params)*scale))

    day2 = int(round(logistic(x[-1]+2, *params)*scale))

    day3 = int(round(logistic(x[-1]+3, *params)*scale))

    show(f'today|day 1|day 2|day 3\n---|---|---|---\n{today}|{day1}|{day2}|{day3}')



def analyze_country(covid, country):

    covid = covid.copy()

    covid = covid.loc[covid['Country/Region'] == country]

    del covid['Lat']

    del covid['Long']

    del covid['Country/Region']

    del covid['Province/State']

    covid['Day'] = [days(date)for date in covid['Date']]

    del covid['Date']

    covid = covid.groupby(['Day']).sum()

    for yName in ['Confirmed', 'Recovered', 'Deaths']:

        analyze_y(covid, yName)
covid = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

covid
most_affected = covid.loc[covid['Date'] == '3/26/20'].groupby('Country/Region').sum()

most_affected = most_affected.loc[most_affected['Confirmed'] > 10000]

most_affected = most_affected.sort_values('Confirmed', ascending=False)

most_affected
countries = ['US', 'China', 'Italy', 'Spain',

             'Germany', 'France','Iran',

             'United Kingdom', 'Switzerland', 'Romania']

for country in countries:

    show(f'## *{country}*')

    analyze_country(covid, country)