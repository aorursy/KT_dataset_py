# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from pylab import *

import matplotlib.pyplot as plt 

import time

from datetime import datetime

import calendar
dailydata = pd.read_csv("../input/fahrzeitensollist2016091120160917.csv")

dailydata=dailydata[:-1]
dailydata["departureTime"] = pd.Series(time.strftime('%H:%M:%S', time.gmtime(x)) for x in dailydata['soll_an_nach'])

dailydata["departureDateTime"] = pd.to_datetime(dailydata["betriebsdatum"] +' ' + dailydata["departureTime"],format="%d.%m.%y %H:%M:%S") 

dailydata["day_departure"] = pd.Series(calendar.day_name[x.weekday()] for x in dailydata['departureDateTime'])
sd_m_dis=dailydata["departureDateTime"].dt.day.value_counts()

sd_m_dis=sd_m_dis.sort_index() 

sd_m_Hr=dailydata["departureDateTime"].dt.hour.value_counts()

sd_m_Hr=sd_m_Hr.sort_index() 
sd_m_mean=sd_m_dis.mean()

print("Month Departures:",sd_m_mean)

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        plt.text(rect.get_x()+rect.get_width()/2., 1.03*height, '%s' % int(height))

        

figure(0)

rects=plt.bar(sd_m_dis.index,sd_m_dis.values)

plt.plot([0,len(sd_m_dis.index)+1],[sd_m_mean,sd_m_mean],"r--")

plt.title("Month Departures")

plt.xlabel("Month")

plt.ylabel("Departures")

plt.grid()

autolabel(rects)
sd_hr_mean=sd_m_Hr.mean()

print("Hour Departures:",sd_hr_mean) 

figure(1)

rects=plt.bar(sd_m_Hr.index,sd_m_Hr.values)

plt.plot([0,len(sd_m_Hr.index)+1],[sd_hr_mean,sd_hr_mean],"r--")

plt.title("Hour Departures")

plt.xlabel("Hour")

plt.ylabel("Departures")

plt.grid()

autolabel(rects) 
dailydata["plan_departureTime"] = pd.Series(time.strftime("%H:%M:%S",time.gmtime(x)) for x in dailydata['ist_an_nach1'])

dailydata["plan_departureDateTime"] = pd.to_datetime(dailydata["betriebsdatum"] +' ' + dailydata["plan_departureTime"],format="%d.%m.%y %H:%M:%S")



def getTimeDifference(TimeStart, TimeEnd):

    timeDiff = TimeEnd - TimeStart

    return round(timeDiff.total_seconds() / 60,0)



departure_plan_diff=[]

for item in dailydata.values:

    item=list(item)   

    departure_plan_diff.append(getTimeDifference(item[35], item[38]))

    

dailydata["departure_plan_diff"] = departure_plan_diff

departure_plan_diff = dailydata["departure_plan_diff"].value_counts()
sd_dept_diff_mean=departure_plan_diff.mean()

print("Diff mean:",sd_dept_diff_mean) 

figure(3)

plt.scatter(departure_plan_diff.index, departure_plan_diff.values, alpha=.1, s=400) 

plt.show()