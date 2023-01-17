# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import relevant libraries

import pandas as pd

from matplotlib import pyplot as plt

import numpy as np

%matplotlib inline
#Import the data

df = pd.read_csv('../input/ebola-outbreak-20142016-complete-dataset/ebola_2014_2016_clean.csv')

df.head()
from datetime import date

import datetime as dt

df['Dates'] = pd.to_datetime(df['Date'])

df['Year']= df.Dates.dt.year

df['Month_name'] = df.Dates.dt.month_name()

df['Day_name'] = df.Dates.dt.day_name()

df['Month'] = df.Dates.dt.month

df["Week"] = df.Dates.dt.week

df['Day_of_year']= df.Dates.dt.dayofyear

#Fetching the data by country (Nigeria)

Nigeria = df.loc[df.Country == 'Nigeria']

Nigeria.head()
d1 = date(2014,8,29)

d2 = date(2016,3,23)

delta = d2-d1

print(delta)
print('The date of  Nigeria data is from', Nigeria.Dates.min(), 'to', Nigeria.Dates.max(),

      ',a total number of', delta)

print('The total number of confirmed cases in Nigeria is', Nigeria['No. of confirmed cases'].sum())

print('The total number of confirmed deaths in Nigeria is', Nigeria['No. of confirmed deaths'].sum())

print('The total number of suspected cases in Nigeria is', Nigeria['No. of suspected cases'].sum())

print('The total number of suspected deaths in Nigeria is', Nigeria['No. of suspected deaths'].sum())

print('The total number of probable cases in Nigeria is', Nigeria['No. of probable cases'].sum())

print('The total number of probable deaths in Nigeria is', Nigeria['No. of probable deaths'].sum())
Nigeria.groupby('Month_name')['No. of confirmed cases', 'No. of confirmed deaths'].sum()
#Months with the highest number of confirmed cases(3)

Nigeria.groupby('Month_name')['No. of confirmed cases'].sum().nlargest(3)
#Months with the highest number of confirmed deaths(3)

Nigeria.groupby('Month_name')['No. of confirmed deaths'].sum().nlargest(3)
#Barcharts showing months with the highest confirmed cases and highest confirmed deaths(3)

plt.subplot(1,2,1)

Nigeria.groupby('Month_name')['No. of confirmed cases'].sum().nlargest(3).plot(kind = 'bar', grid = True)

plt.title('Confirmed cases (3)')

plt.xlabel('Months')

plt.ylabel('No. of probable cases')

plt.subplot(1,2,2)

Nigeria.groupby('Month_name')['No. of confirmed deaths'].sum().nlargest(3).plot(kind = 'bar', grid = True,

                                                                       color = 'red')

plt.title('Confirmed deaths (3)')

plt.xlabel('Months')

plt.ylabel('No. of probable deaths')

plt.tight_layout()

plt.show()
#Months with the highest number of suspected cases(3)

Nigeria.groupby('Month_name')['No. of suspected cases'].sum().nlargest(3)
#Months with the highest number of suspected deaths(3)

Nigeria.groupby('Month_name')['No. of suspected deaths'].sum().nlargest(3)
#Barchart showing months with the highest number of suspected cases(3)

plt.subplot(1,2,1)

Nigeria.groupby('Month_name')['No. of suspected cases'].sum().nlargest(3).plot(kind = 'bar', grid = True)

plt.title('Suspected cases (3)')

plt.xlabel('Months')

plt.ylabel('No. of suspected cases')

plt.show()
#Barcharts showing months with the highest number of probable cases and highest number of probable deaths(3)

plt.subplot(1,2,1)

Nigeria.groupby('Month_name')['No. of probable cases'].sum().nlargest(3).plot(kind = 'bar', grid = True)

plt.title('Probable cases (3)')

plt.xlabel('Months')

plt.ylabel('No. of probable cases')

plt.subplot(1,2,2)

Nigeria.groupby('Month_name')['No. of probable deaths'].sum().nlargest(3).plot(kind = 'bar', grid = True,

                                                                       color = 'red')

plt.title('Probable deaths (3)')

plt.xlabel('Months')

plt.ylabel('No. of probable deaths')

plt.tight_layout()

plt.show()
#Week with the highest number of confirmed cases(3)

Nigeria.groupby('Week')['No. of confirmed cases'].sum().nlargest(3)
#Week with the highest number of confirmed deaths(3)

Nigeria.groupby('Week')['No. of confirmed deaths'].sum().nlargest(3)
#Barcharts showing the week with the highest number of confirmed cases and highest number of confirmed deaths(3)

plt.subplot(1,2,1)

Nigeria.groupby('Week')['No. of confirmed cases'].sum().nlargest(3).plot(kind = 'bar', grid = True)

plt.title('Confirmed cases (3)')

plt.xlabel('Week')

plt.ylabel('No. of confirmed cases')

plt.subplot(1,2,2)

Nigeria.groupby('Week')['No. of confirmed deaths'].sum().nlargest(3).plot(kind = 'bar', grid = True,

                                                                       color = 'red')

plt.title('Confirmed deaths (3)')

plt.xlabel('Week')

plt.ylabel('No. of confirmed deaths')

plt.tight_layout()

plt.show()
#Week with the highest number of suspected cases(3)

Nigeria.groupby('Week')['No. of suspected cases'].sum().nlargest(3)
#Week with the highest number of suspected deaths(3)

Nigeria.groupby('Week')['No. of suspected deaths'].sum().nlargest(3)
#Barchart showing the weeks with the highest number suspected cases (3)

plt.subplot(1,2,1)

Nigeria.groupby('Week')['No. of suspected cases'].sum().nlargest(3).plot(kind = 'bar', grid = True)

plt.title('Suspected cases (3)')

plt.xlabel('Week')

plt.ylabel('No of suspected cases')

plt.show()
#Week with the highest number of probable cases(3)

Nigeria.groupby('Week')['No. of probable cases'].sum().nlargest(3)
#Week with the highest number of probable deaths(3)

Nigeria.groupby('Week')['No. of probable deaths'].sum().nlargest(3)
#Barcharts showing the weeks with the highest number of probable cases and highest number of probable deaths(3)

plt.subplot(1,2,1)

Nigeria.groupby('Week')['No. of probable cases'].sum().nlargest(3).plot(kind = 'bar', grid = True)

plt.title('Probable cases (3)')

plt.xlabel('Week')

plt.ylabel('No of probable cases')

plt.subplot(1,2,2)

Nigeria.groupby('Week')['No. of probable deaths'].sum().nlargest(3).plot(kind = 'bar', grid = True,

                                                                       color = 'red')

plt.title('Probleble deaths (3)')

plt.xlabel('Week')

plt.ylabel('No of probable deaths')

plt.tight_layout()

plt.show()