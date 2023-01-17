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
import datetime as dt

from datetime import date

df['Dates'] = pd.to_datetime(df['Date'])

df['Year']= df.Dates.dt.year

df['Month_name'] = df.Dates.dt.month_name()

df['Day_name'] = df.Dates.dt.day_name()

df['Month'] = df.Dates.dt.month

df["Week"] = df.Dates.dt.week

df['Day_of_year']= df.Dates.dt.dayofyear

d0 = date(2014,8,29)

d1 = date(2016,3,23)

delta = d1-d0

print(delta)
df.head()
#Number of rows and columns

df.shape

df.groupby('Country')['No. of confirmed cases', 'No. of confirmed deaths'].sum()
print('The date of the data is from', df.Dates.min(), 'to', df.Dates.max(),',a total amount of'

      , delta)

print('The total number of confirmed cases is', df['No. of confirmed cases'].sum())

print('The total number of confirmed deaths is', df['No. of confirmed deaths'].sum())

print('The total number of suspected cases is', df['No. of suspected cases'].sum())

print('The total number of suspected deaths is', df['No. of suspected deaths'].sum())

print('The total number of probable cases is', df['No. of probable cases'].sum())

print('The total number of probable deaths is', df['No. of probable deaths'].sum())
#Countries with the highest number confirmed cases (3)

df.groupby('Country')['No. of confirmed cases'].sum().nlargest(3)
#Countries with the highest number confirmed deaths (3)

df.groupby('Country')['No. of confirmed deaths'].sum().nlargest(3)
#Barcharts showing Countries with the highest number of confirmed cases and highest number confirmed deaths(3)

plt.subplot(1,2,1)

df.groupby('Country')['No. of confirmed cases'].sum().nlargest(3).plot(kind = 'bar', grid = True)

plt.title('Confirmed cases (3)')

plt.xlabel('Countries')

plt.ylabel('No of confirmed cases')

plt.subplot(1,2,2)

df.groupby('Country')['No. of confirmed deaths'].sum().nlargest(3).plot(kind = 'bar', grid = True,

                                                                       color = 'red')

plt.title('Confirmed deaths (3)')

plt.xlabel('Countries')

plt.ylabel('No of confirmed deaths')

plt.tight_layout()

plt.show()
#Countries with the highest number of suspected cases (3)

df.groupby('Country')['No. of suspected cases'].sum().nlargest(3)
#Countries with the highest number of suspected deaths (3)

df.groupby('Country')['No. of suspected deaths'].sum().nlargest(3)
#Barcharts showing Countries with the highest number of suspected cases and highest number of suspected deaths(3)

plt.subplot(1,2,1)

df.groupby('Country')['No. of suspected cases'].sum().nlargest(3).plot(kind = 'bar', grid = True)

plt.title('Suspected cases (3)')

plt.xlabel('Countries')

plt.ylabel('No of suspected cases')

plt.subplot(1,2,2)

df.groupby('Country')['No. of suspected deaths'].sum().nlargest(3).plot(kind = 'bar', grid = True,

                                                                       color = 'red')

plt.title('Suspected deaths (3)')

plt.xlabel('Countries')

plt.ylabel('No of suspected deaths')

plt.tight_layout()

plt.show()
#Countries with the highest number of probable cases(3)

df.groupby('Country')['No. of probable cases'].sum().nlargest(3)
#Countries with the highest number of probable deaths(3)

df.groupby('Country')['No. of probable deaths'].sum().nlargest(3)
#Barcharts showing Countries with the highest number of probable cases and highest number of probable deaths(3)

plt.subplot(1,2,1)

df.groupby('Country')['No. of probable cases'].sum().nlargest(3).plot(kind = 'bar', grid = True)

plt.title('Probable cases (3)')

plt.xlabel('Countries')

plt.ylabel('No of probable cases')

plt.subplot(1,2,2)

df.groupby('Country')['No. of probable deaths'].sum().nlargest(3).plot(kind = 'bar', grid = True,

                                                                       color = 'red')

plt.title('Probable deaths (3)')

plt.xlabel('Countries')

plt.ylabel('No of probable deaths')

plt.tight_layout()

plt.show()
#Countries with the lowest number of confirmed cases(3)

df.groupby('Country')['No. of confirmed cases'].sum().nsmallest(3)
#Countries with the least number of confirmed deaths(3)

df.groupby('Country')['No. of confirmed deaths'].sum().nsmallest(3)
#Countries with the least number of suspected cases(3)

df.groupby('Country')['No. of suspected cases'].sum().nsmallest(3)
#Countries with the least number of suspected deaths(3)

df.groupby('Country')['No. of suspected deaths'].sum().nsmallest(3)
#Countries with the least number of probable cases(3)

df.groupby('Country')['No. of probable cases'].sum().nsmallest(3)
#Countries with the least number of probable deaths(3)

df.groupby('Country')['No. of probable deaths'].sum().nsmallest(3)