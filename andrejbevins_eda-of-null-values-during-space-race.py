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
## import libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns
## read in the dataset



sd = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')
## look at the dataset 



sd.head()
## changing datum to datetime



sd['Datum'] = pd.to_datetime(sd['Datum'], utc=True)
## rename rocket, comapny name, Status rocket, and Status Mission



sd = sd.rename(columns = {' Rocket' : 'Rocket_Price'})

sd = sd.rename(columns = {'Company Name': 'Company_Name'})

sd = sd.rename(columns = {'Status Rocket': 'Status_Rocket'})

sd = sd.rename(columns = {'Status Mission': 'Status_Mission'})
## find missing values



sd.isnull().sum()
##fill missing values for Rocket_Price with 0



sd = sd.fillna(0)
## change datatype for rocket_price



sd['Rocket_Price'] = pd.to_numeric(sd['Rocket_Price'], errors = 'coerce')
## DROP UNWANTED COLUMNS



sd = sd.drop(sd.columns[[0,1,5]], axis = 1)
## add new columns



sd['Day'] = sd['Datum'].dt.day_name()

sd['Month'] = sd['Datum'].dt.month_name()

sd['Year'] = sd['Datum'].dt.year



sd['Exact_Location'] = sd['Location'].apply(lambda Location:Location.split(',')[-1])
## fix index starting at 0 to 1



sd.index = np.arange(1, len(sd) + 1)
## check new dataset



sd.head(20)
## create null and non null datasets



sd_null = sd[sd['Rocket_Price']== 0]

sd_valid = sd[sd['Rocket_Price'] != 0]



## launches by country 



plt.figure(figsize = (17,17))

figure_country_null = sns.countplot(y='Exact_Location', data = sd, order = sd['Exact_Location'].value_counts().index, 

                             palette = 'Blues_r')

plt.title('Number of Launches Per Country', fontsize = 35)

plt.xlabel('Frequency', fontsize = 30)

plt.ylabel('Location', fontsize = 30)

plt.show()
## ratio of valid rocket price records to total records 



country_full = sd['Exact_Location'].value_counts()

country_null = sd_null['Exact_Location'].value_counts()

country_valid =sd_valid['Exact_Location'].value_counts()

print(country_valid/country_full)
## ratio of null rocket price to total records 



print(country_null/country_full)
## recorded launches with absent rocket price information



plt.figure(figsize = (17,17))

figure_country_null = sns.countplot(y='Exact_Location', data = sd_null, order = sd_null['Exact_Location'].value_counts().index, 

                             palette = 'Blues_r')

plt.title('Number of Launches Without Rocket Price Record', fontsize = 35)

plt.ylabel('Location', fontsize = 30)

plt.xlabel('Frequency', fontsize = 30)

plt.show()
## recorded launches with valid rocket price information 



plt.figure(figsize = (15,17))

figure_country_nonnull = sns.countplot(y='Exact_Location', data = sd_valid, order = sd_valid['Exact_Location'].value_counts().index, 

                                 palette= 'BuGn_r')

plt.title('Number of Launches With Rocket Price Record', fontsize = 35)

plt.ylabel('Number of Launches Into Space', fontsize = 22)

plt.xlabel('Location', fontsize = 30)

plt.show()
## Years with most frequent missing values



plt.figure(figsize = (16,24))

figure_year_null = sns.countplot(y='Year', data = sd_null, order = sd_null['Year'].value_counts().index, 

                             palette = 'BuGn_r')

plt.title('Years With Most Null Rocket Price Values', fontsize = 30)

plt.ylabel('Year', fontsize = 30)

plt.xlabel('Frequency', fontsize = 30)

plt.show()
## Years with most valid values



plt.figure(figsize = (20,24))

figure_year_nonnull = sns.countplot(y='Year', data = sd_valid, order = sd_valid['Year'].value_counts().index, 

                             palette = 'Blues_r')

plt.title('Years With Most Non-Null Rocket Price Values', fontsize = 30)

plt.ylabel('Year', fontsize = 30)

plt.xlabel('Frequency', fontsize = 30)

plt.show()
## finding ratios 



year_null = sd_null['Year'].value_counts()

year_valid = sd_valid['Year'].value_counts()

year_origional = sd['Year'].value_counts()



ratio = year_null/year_valid

null_less_than_valid = ratio[ratio < 1]

null_greater_than_valid = ratio[ratio > 1]

## year percentage of null to valid rocket price where null is less than valid (top 10)



null_less_than_valid = ratio[ratio < 1]

null_less_than_valid[:10]
## year percentage of null to valid rocket price where null is greater than valid (top 10)



null_greater_than_valid = ratio[ratio > 1]

null_greater_than_valid[:10]
## launches by year 



plt.figure(figsize = (20,20))

year_launches = sns.countplot(y = 'Year', data= sd, order = sd['Year'].value_counts().index, palette = 'Blues_r')

plt.title('Launches per Year', fontsize = 30)

plt.ylabel('Year', fontsize = 30)

plt.xlabel('Number of Launches', fontsize = 30)

plt.show()
sd['Year'].value_counts()[:10]
## null values by company



plt.figure(figsize = (15,25))

figure_companyname_null = sns.countplot(y='Company_Name', data = sd_null, order = sd_null['Company_Name'].value_counts().index, 

                             palette = 'Blues_r', )

plt.title('Company Launches Without Rocket Price Record', fontsize = 35)

plt.ylabel('Company', fontsize = 30)

plt.xlabel('Frequency', fontsize = 30)

plt.show()
## valid values by company



plt.figure(figsize = (15,20))

figure_companyname_null = sns.countplot(y='Company_Name', data = sd_valid, order = sd_valid['Company_Name'].value_counts().index, 

                             palette = 'Blues_r', )

plt.title('Company Launches With Rocket Price Record', fontsize = 35)

plt.ylabel('Company', fontsize = 30)

plt.xlabel('Frequency', fontsize = 30)

plt.show()
plt.figure(figsize = (20, 12))

launches_by_day = sns.countplot(x='Month', data=sd_null, order = sd_null['Month'].value_counts().index, palette = 'Blues_r')

plt.title('Launch Month With Null Record', fontsize = 25)

plt.xlabel('Month', fontsize= 25)

plt.ylabel('Frequency', fontsize = 25)

plt.show()
plt.figure(figsize = (20, 11))

launches_by_day = sns.countplot(x='Month', data=sd_valid, order = sd_valid['Month'].value_counts().index, palette = 'Blues_r')

plt.title('Launch Month With Valid Record', fontsize = 25)

plt.ylabel('Frequency', fontsize= 25)

plt.xlabel('Month', fontsize = 25)

plt.show()
plt.figure(figsize = (10, 10))

launches_by_day = sns.countplot(x='Day', data=sd_valid, order = sd_valid['Day'].value_counts().index, palette = 'Blues_r')

plt.title('Launch Day With Valid Rocket Price', fontsize = 25)

plt.xlabel('Day', fontsize = 25)

plt.ylabel('Frequency', fontsize = 25)

plt.show()
plt.figure(figsize = (10, 10))

launches_by_day = sns.countplot(x='Day', data=sd_null, order = sd_null['Day'].value_counts().index, palette = 'Blues_r')

plt.title('Launch Day With Null Rocket Price', fontsize = 25)

plt.xlabel('Day', fontsize = 25)

plt.ylabel('Frequency', fontsize = 25)

plt.show()