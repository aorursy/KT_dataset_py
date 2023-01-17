%matplotlib inline



import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt



matplotlib.style.use('ggplot')
ato_file = "../input/ato.csv"

ato = pd.read_csv(ato_file, parse_dates=False)

ato
ato_month_file = "../input/ato_month.csv"

ato_month = pd.read_csv(ato_month_file)



import datetime

import calendar



def add_months(sourcedate, months):

    month = sourcedate.month - 1 + months

    year = sourcedate.year + month // 12

    month = month % 12 + 1

    day = min(sourcedate.day, calendar.monthrange(year,month)[1])

    return datetime.date(year, month, day)



ato_month['month'] = ato_month['month'].apply(lambda d: pd.datetime.strptime(d, '%m/%d/%y')).dt.date

ato_month['month'] = ato_month['month'].apply(lambda d: add_months(d,1))





ato_month.head()
import matplotlib.dates as mdates

import datetime



years = mdates.YearLocator()   # every year

months = mdates.MonthLocator()  # every month

yearsFmt = mdates.DateFormatter('%Y')



fig, ax = plt.subplots(2,1,figsize=(15,10))



ax[0].set_title('LIA by Month / Втрати за місяць\nVertical dash lines are the cease fire treaties')

ax[1].set_title('LIA by Year / Втрати за рік')



ax[0].plot(ato_month['month'], ato_month['losses'])

ax[1].bar(ato['year'], ato['losses'])





# format the ticks

ax[0].xaxis.set_major_locator(years)

ax[0].xaxis.set_major_formatter(yearsFmt)

ax[0].xaxis.set_minor_locator(months)



ax[0].axvline(x = datetime.date(2014, 9, 5), linestyle = ':', color='gray')

ax[0].axvline(x = datetime.date(2015, 2, 11), linestyle = ':', color='gray')



plt.xticks(rotation=90)



plt.show()