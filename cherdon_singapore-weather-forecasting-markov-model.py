import datetime
import pandas as pd
import numpy as np
import time
weatherdata = pd.read_csv("../input/WEATHERDATA_2016-2018.csv", encoding="ISO-8859-1")
weatherdata.tail()
weatherdata['Date'] = pd.to_datetime(weatherdata[['Year', 'Month', 'Day']])
data = weatherdata.iloc[:, 13:19]
data.tail()
def previous_date(date, n_back):
    last = pd.to_datetime(date) - datetime.timedelta(days=n_back)
    return last

def attributes(date, data):
    attributes = data.loc[data['Date'] == date]
    return attributes

class date_attributes:
    def __init__(self, date, data):
        self.monthlyErr = float(attributes(previous_date(date, 365), data)['MonthlyErr'])
def weatherForecast(date, data):
    today = data.iloc[-1, :]['Date']
    date = datetime.datetime.strptime(date, "%Y-%m-%d")
    totalDays = (date - today).days
    todayRain = data.iloc[-1, :]['Rainy']

    i = 1
    while i <= totalDays:
        day = date_attributes(today, data)
        err = day.monthlyErr
        transitionName = [['11', '10'], ['00', '01']]
        transitionMatrix = [[1-err, err], [1-err, err]]
        if todayRain == 1:
            change = np.random.choice(transitionName[0], replace=True, p=transitionMatrix[0])
            if change == '11':
                pass
            else:
                todayRain = 0
        if todayRain == 0:
            change = np.random.choice(transitionName[1], replace=True, p=transitionMatrix[1])
            if change == '00':
                pass
            else:
                todayRain = 1
        today += datetime.timedelta(days=1)
        if todayRain == 1:
            print(str(today) + ' rainy where P(weather changes)= ' + str(err))
        else:
            print(str(today) + ' dry where P(weather changes)= ' + str(err))
        i += 1
        time.sleep(0.2)
date = "2018-12-7"
weatherForecast(date, data)