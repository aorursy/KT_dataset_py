# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd
###covid_19_data

# save filepath to variable for easier access

covid19_file_path = '../input/novel-corona-virus-2019-dataset/covid_19_data.csv'

# read the data and store data in DataFrame

covid19_data = pd.read_csv(covid19_file_path) 

# print a summary of the dat

covid19_data.describe()
##time_series_covid_19_confirmed

# save filepath to variable for easier access

covid19_confirmed_file_path = '../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv'

# read the data and store data in DataFrame

covid19_confirmed_data = pd.read_csv(covid19_confirmed_file_path) 

# print a summary of the dat

covid19_confirmed_data.describe()
##time_series_covid_19_deaths

# save filepath to variable for easier access

covid19_deaths_file_path = '../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv'

# read the data and store data in DataFrame

covid19_deaths_data = pd.read_csv(covid19_deaths_file_path) 

# print a summary of the dat

covid19_deaths_data.describe()
##time_series_covid_19_recovered

##garren

# save filepath to variable for easier access

covid19_recovered_file_path = '../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv'

# read the data and store data in DataFrame

covid19_recovered_data = pd.read_csv(covid19_recovered_file_path) 

# print a summary of the dat

covid19_recovered_data.describe()
##isolate the US casees of COVID deaths over the past 5 days

##garren

df = pd.DataFrame(covid19_deaths_data, columns=['Country/Region','Province/State','3/16/20','3/17/20','3/18/20','3/19/20','3/20/20'],)

##df['Country/Region']

df.loc[df['Country/Region'] == 'US']
##isolate illinois COVID deaths over the past 5 days

##garren

df = pd.DataFrame(covid19_deaths_data, columns=['Country/Region','Province/State','3/16/20','3/17/20','3/18/20','3/19/20','3/20/20'],)

##df['Country/Region']

df.loc[(df['Country/Region'] == 'US') & (df['Province/State']=='Illinois')]

##isolate illinois COVID confirmed over the past 5 days

##garren

df = pd.DataFrame(covid19_confirmed_data, columns=['Country/Region','Province/State','3/16/20','3/17/20','3/18/20','3/19/20','3/20/20'],)

##df['Country/Region']

df.loc[(df['Country/Region'] == 'US') & (df['Province/State']=='Illinois')]



##Growth rate calculation:  Present-Past/Past
##isolate US COVID confirmed over the past 5 days

##garren

df = pd.DataFrame(covid19_confirmed_data, columns=['Country/Region','Province/State','3/16/20','3/17/20','3/18/20','3/19/20','3/20/20'],)

df.loc[(df['Country/Region'] == 'US')].sum()
###merge datasets experimental

##garren

#pd.merge(covid19_deaths_data,covid19_recovered_data, on='Country/Region')




