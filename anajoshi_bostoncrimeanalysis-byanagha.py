# -*- coding: utf-8 -*-

"""

Content for BOSTON Crime Analysis

    0. Clear Memory

    1. Import

    2. Read data

    3. Function

    4. Explore data

    5. Visualization

    6. Enhancement (Based on feedback received from user community)



Objective: 

    Analyse the Boston Crime Data and get some insight into what types of Crimes were getting reported during     

    the years 2015 to 2018 and also note the observations for next actionable steps in Boston 

    Police Depratment in its Districts.



"""

# 0 Clear memory

%reset -f
# 1.1 Call data manipulation libraries

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd         # linear algebra

import numpy as np          # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# 1.2 Plotting libraries to plot feature importance

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.compose import ColumnTransformer as ct

from sklearn.preprocessing import StandardScaler as ss

from sklearn.preprocessing import OneHotEncoder as ohe

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier as dt



from sklearn import linear_model

import statsmodels.api as sm
# 2.0 Set working directory and read file

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

print(os.getcwd())

os.listdir()

pd.options.display.max_columns = 300
# 2.1 Read train/test files

#data = pd.read_csv("crime.csv")

data = pd.read_csv('../input/crime.csv', encoding='latin-1')
#3 Functions



# 3.1 Print 5 Rows for any column

def print_rows(name_column):

    return data[name_column][0:5]



# 3.2 Get Details of the Column

def describe_column(name_column):

    return data[name_column].describe()
# 4 Explore data

# 4.1 Shape

data.shape              # 327820, 17
# 4.2 Columns

data.columns 
# 4.3 Is Null

data.isnull().sum()
# 4.4 Head gives the first 5 Rows

data.head()
# 4.5 Tail gives last 5 rows

data.tail()
# 4.6 Information

data.info
# 4.7 Code for Offence Group & its Columns

print_rows('OFFENSE_CODE_GROUP')
describe_column('OFFENSE_CODE_GROUP')
# 4.8 Code for Offence Group Description & its Columns

print_rows('OFFENSE_DESCRIPTION')
describe_column('OFFENSE_DESCRIPTION')
# 4.9 Code for Ploice District & its Columns

print_rows('DISTRICT')
describe_column('DISTRICT')
data['DISTRICT'].unique()
# 4.10 Code for Reporting Area & its Columns

print_rows('REPORTING_AREA')
describe_column('REPORTING_AREA')
# 4.11 Code for Crime with Shooting Yes / Nan & its Columns

print_rows('SHOOTING')
describe_column('SHOOTING')
# 4.12 Code for Crime Occoured on Date & its Columns

print_rows('OCCURRED_ON_DATE')
data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'])
describe_column('OCCURRED_ON_DATE')
# 4.13 Code for Year of crime & its Columns

print_rows('YEAR')
describe_column('YEAR')
data['YEAR'].unique()
# 4.14 Code for Month of crime & its Columns

print_rows('MONTH')
describe_column('MONTH')
data['MONTH'].unique()
describe_column('DAY_OF_WEEK')
data['DAY_OF_WEEK'].unique()
# 4.16 Code for Hour of crime & its Columns

print_rows('HOUR')
describe_column('HOUR')
data['HOUR'].unique()
# 4.17 Code for Hour of crime & its Columns

print_rows('UCR_PART')
describe_column('UCR_PART')
data['UCR_PART'].unique()
# 4.18 Code for Hour of crime & its Columns

print_rows('STREET')
describe_column('STREET')
data['STREET'].unique()
# 4.19 Code for Location, Latitude & Logitude & its Columns

print_rows('Lat')
describe_column('Lat')
# 5 Data Visualization

# 5.1 District

plt.figure(figsize=(16,8))

data['DISTRICT'].value_counts().plot.bar()

plt.title('BOSTON: District wise Crimes')

plt.ylabel('Number of Crimes')

plt.xlabel('Police District')

plt.show()

# Maxium number of Crimes observed in Police District B2 in Boston.

# 5.1.1 YEARWISE breakup of Crimes by District

sns.catplot(x="DISTRICT",       # Variable whose distribution (count) is of interest

            hue="MONTH",      # Show distribution, pos or -ve split-wise

            col="YEAR",       # Create two-charts/facets, gender-wise

            data=data,

            kind="count")

#Crime rates are consistent in the districts B2, C11 & D4 across the 4 years
# 5.1.2 For individual years: 2015

#data['SHOOTING'].unique()

plt.figure(figsize=(16,8))

data['DISTRICT'].loc[data['YEAR']==2015].value_counts().plot.bar()

plt.title('BOSTON: Police District Wise Crimes in 2015')

plt.show()

# Maxium number of Crimes observed in Police District B2 in Boston.

# 5.1.3 For individual years: 2016

plt.figure(figsize=(16,8))

data['DISTRICT'].loc[data['YEAR']==2016].value_counts().plot.bar()

plt.title('BOSTON: Police District Wise Crimes in 2016')

plt.show()

# Maxium number of Crimes observed in Police District B2 in Boston.
# 5.1.4 For individual years: 2017

plt.figure(figsize=(16,8))

data['DISTRICT'].loc[data['YEAR']==2017].value_counts().plot.bar()

plt.title('BOSTON: Police District Wise Crimes in 2017')

plt.show()

# Maxium number of Crimes observed in Police District B2 in Boston.

# 5.1.5 For individual years: 2018

plt.figure(figsize=(16,8))

data['DISTRICT'].loc[data['YEAR']==2018].value_counts().plot.bar()

plt.title('BOSTON: Police District Wise Crimes in 2018')

plt.show()

# Maxium number of Crimes observed in Police District B2 in Boston.

# 5.2 Offence Code Group

plt.figure(figsize=(16,8))

data['OFFENSE_CODE_GROUP'].value_counts().plot.bar()

plt.title('BOSTON: Crime Offence Code Groups')

plt.show()

# Motor Vehicle Accident Response tops in the list.
# 5.3 Number of Crimes reported in Boston Each Year

plt.figure(figsize=(16,8))

sns.countplot(x='YEAR', data = data)

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Year wise # of Crimes Reported')

plt.show()

# Maxium number of Crimes is observed in 2017
# 5.4 Top 10

# 5.4.1 Top 10 Crime District Locations

plt.figure(figsize=(16,8))

top10cloc = data.groupby('DISTRICT')['INCIDENT_NUMBER'].count().sort_values(ascending=False)

top10cloc = top10cloc [:10]

top10cloc.plot(kind='bar', color='green')

plt.ylabel('Number of Crimes')

plt.xlabel("POLICE DISTRICTS")

plt.title('BOSTON: Top 10 Crime District Locatins')

#plt.savefig("C:\\Users\\bk42969\\Desktop\\BigData\\Top10CrimeDistrictLocation.png")

plt.show()

# Maxium number of Crimes observed in Police District B2 in Boston.
# 5.4.2 Top 10 Types of Crime

plt.figure(figsize=(16,8))

top10ctype = data.groupby('OFFENSE_CODE_GROUP')['INCIDENT_NUMBER'].count().sort_values(ascending=False)

top10ctype = top10ctype [:10]

top10ctype.plot(kind='bar', color='blue')

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Top 10 Types of Crime')

plt.show()

# Motor Vehicle Accident Response tops in the list.
# 5.5 Shooting involved in Crime

plt.figure(figsize=(16,8))

data.groupby(['SHOOTING'])['INCIDENT_NUMBER'].count().plot(kind = 'bar')

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Shooting Crimes Reported')

plt.show()

# 0.305% of Boston Crimes are due to Shooting.
# 5.6 When do serious crimes occur?

#We can consider patterns across several different time scales: hours of the day, days of the week, and months of the year.

# 5.6.1 Number of Crimes reported at Hour during the Day

plt.figure(figsize=(16,8))

sns.countplot(x='HOUR', data = data)

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Hour wise # of Crimes Reported')

plt.show()

# Crimes are observed Least in the Early Hours of the Morning. 

# 5.6.2 Comparing Weekly crimes

plt.figure(figsize=(16,8))

data.groupby(['DAY_OF_WEEK'])['INCIDENT_NUMBER'].count().plot(kind = 'bar')

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Crimes Reports - Day of the Week')

plt.show()

# Observed that Sunday has least number of Crimes
# 5.6.3 Comparing crimes during months.

plt.figure(figsize=(16,8))

data.groupby(['MONTH'])['INCIDENT_NUMBER'].count().plot(kind = 'bar')

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Month wise Crimes')

plt.show()

# Maximum number of Crimes observed in July, Aug & Sep Months.

plt.figure(figsize=(16,8))

sns.countplot(x='UCR_PART', data = data)

plt.ylabel('Number of Crimes')

plt.title('BOSTON: UCR Type of Crimes Reported')

plt.show()

# Top in the list is "Part 3" type of UCR Crimes.

# 5.8 Boston Crimes: Year Wise Police District Wise Crimes 

groups = data['DISTRICT'].unique()

n_groups = len(data['DISTRICT'].unique())-1



#fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.2

opacity= 0.8

plt.figure(figsize=(16,8))



df = data[['DISTRICT','YEAR']]



df_2015 = df.loc[(df['YEAR'] == 2015)]

df_2016 = df.loc[(df['YEAR'] == 2016)]

df_2017 = df.loc[(df['YEAR'] == 2017)]

df_2018 = df.loc[(df['YEAR'] == 2018)]



crimes_2015 = df_2015['DISTRICT'].value_counts()

crimes_2016 = df_2016['DISTRICT'].value_counts()

crimes_2017 = df_2017['DISTRICT'].value_counts()

crimes_2018 = df_2018['DISTRICT'].value_counts()



bar1 = plt.bar(index, crimes_2015, bar_width, alpha = opacity, color = 'b', label = '2015')

bar2 = plt.bar(index + bar_width, crimes_2016, bar_width, alpha = opacity, color = 'c', label = '2016')

bar3 = plt.bar(index+ bar_width+ bar_width, crimes_2017, bar_width, alpha = opacity, color = 'r', label = '2017')

bar4 = plt.bar(index+ bar_width+ bar_width+ bar_width, crimes_2018, bar_width, alpha = opacity, color = 'y', label = '2018')





plt.ylabel('Number of Crimes')

plt.xlabel("POLICE DISTRICTS")

plt.title('BOSTON: Police District Wise # of Yearly Crimes')

plt.xticks(index + bar_width, groups)

plt.legend()

plt.show()



# 6.1 BOSTON: Cummilative Month wise Crimes

plt.figure(figsize=(16,8))

data.groupby(['MONTH'])['INCIDENT_NUMBER'].count().plot(marker = 'o')

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Cummilative Month wise Crimes')

plt.show()
# 6.2 Comparing crimes per month from 2015 to 2018

plt.figure(figsize=(16,8))

data.groupby(['MONTH', 'YEAR'])['INCIDENT_NUMBER'].count().unstack().plot(kind = 'bar')

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Month wise Crimes During 2015 to 2018')

plt.show()
# 6.3 Comparing crimes per Week from 2015 to 2018

plt.figure(figsize=(16,8))

data.groupby('DAY_OF_WEEK')['INCIDENT_NUMBER'].count().plot(marker = 'o')

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Day of the Week wise Crimes During 2015 to 2018')

plt.show()
# 6.4 Top 10 locations of Crime

plt.figure(figsize=(16,8))

top10loc = data.groupby('STREET')['INCIDENT_NUMBER'].count().sort_values(ascending=False)

top10loc = top10loc [:10]

top10loc.plot(kind='bar', color='blue')

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Top 10 locations of Crime')

plt.show()
# 6.5 Year-wise Crimes due to Shooting.

plt.figure(figsize=(16,8))

data.groupby('SHOOTING')['INCIDENT_NUMBER'].count().plot(kind = 'bar')

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Year-wise Crimes due to Shooting.')

plt.show()
# 6.6 Monthly trend of top 5 crimes

plt.figure(figsize=(16,8))

top5crimes = data[(data['OFFENSE_CODE_GROUP'] == 'Motor Vehicle Accident Response') |

        (data['OFFENSE_CODE_GROUP'] == 'Larceny') |

        (data['OFFENSE_CODE_GROUP'] == 'Medical Assistance') |

        (data['OFFENSE_CODE_GROUP'] == 'Investigate Person') |

        (data['OFFENSE_CODE_GROUP'] == 'Others')]

top5crimes = top5crimes.pivot_table(values = 'INCIDENT_NUMBER', index = 'MONTH', columns = 'YEAR', aggfunc = np.size)

top5crimes

sns.heatmap(top5crimes)

plt.xlabel("Month")

plt.ylabel('Year')

plt.title('BOSTON: Heat Map of Month wise Yearly Crimes')

plt.show()
# 6.7 Lets get a clear picture on this.

plt.figure(figsize=(16,8))

data.groupby(['MONTH', 'YEAR'])['INCIDENT_NUMBER'].count().unstack().plot(marker = 'o')

plt.xticks(np.arange(12), 'MONTH')

plt.ylabel('Number of Crimes')

plt.title('BOSTON: Month wise Crimes During 2015 to 2018')

plt.show()
# 6.8 Hours of Crime: Heat Map

# 6.8.a. Heat Map

plt.figure(figsize=(16,8))

tophours = data[(data['OFFENSE_CODE_GROUP'] == 'Motor Vehicle Accident Response') |

        (data['OFFENSE_CODE_GROUP'] == 'Larceny') |

        (data['OFFENSE_CODE_GROUP'] == 'Medical Assistance') |

        (data['OFFENSE_CODE_GROUP'] == 'Investigate Person') |

        (data['OFFENSE_CODE_GROUP'] == 'Others')]

data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'])

tophours = tophours.pivot_table(values = 'INCIDENT_NUMBER', index = 'OFFENSE_CODE_GROUP', columns = data['OCCURRED_ON_DATE'].dt.hour , aggfunc = np.size)

tophours

sns.heatmap(tophours)

plt.xlabel('Hours of the Day')

plt.ylabel('Type of Crime')

plt.title('BOSTON: Heat Map of Hour wise Crimes')

plt.show()
# 6.8.b. Line Graph

plt.figure(figsize=(128,128))

data.groupby([data['OCCURRED_ON_DATE'].dt.hour, 'OFFENSE_CODE_GROUP'])['INCIDENT_NUMBER'].count().unstack().plot(marker = 'o')

plt.ylabel('Number of Crimes')

plt.xlabel('Hours ofthe Day')

plt.xticks(np.arange(24))

plt.title('BOSTON: Hourly Crimes During 2015 to 2018')

plt.show()
# 6.9 Monthly trend in Top 5 types of crime

# 6.9.a. Heat Map

plt.figure(figsize=(16,8))

topmonths = data[(data['OFFENSE_CODE_GROUP'] == 'Motor Vehicle Accident Response') |

        (data['OFFENSE_CODE_GROUP'] == 'Larceny') |

        (data['OFFENSE_CODE_GROUP'] == 'Medical Assistance') |

        (data['OFFENSE_CODE_GROUP'] == 'Investigate Person') |

        (data['OFFENSE_CODE_GROUP'] == 'Others')]

data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'])

topmonths = topmonths.pivot_table(values = 'INCIDENT_NUMBER', index = 'OFFENSE_CODE_GROUP', columns = 'MONTH', aggfunc = np.size)

topmonths

sns.heatmap(topmonths)

plt.xlabel('Months')

plt.ylabel('Type of Crime')

plt.title('BOSTON: Heat Map of Month wise Crimes')

plt.show()
# 6.9.b. Line Graph

plt.figure(figsize=(128,32))

data.groupby(['MONTH', 'OFFENSE_CODE_GROUP'])['INCIDENT_NUMBER'].count().unstack().plot(marker = 'o')

plt.ylabel('Number of Crimes')

plt.xlabel('Months')

plt.xticks(np.arange(12), 'MONTH')

plt.title('BOSTON: Monthly Crimes During 2015 to 2018')

plt.show()
# 6.10 Weekly trend of top 5 crimes

# 6.10.a. Heat Map

plt.figure(figsize=(16,8))

topweeks = data[(data['OFFENSE_CODE_GROUP'] == 'Motor Vehicle Accident Response') |

        (data['OFFENSE_CODE_GROUP'] == 'Larceny') |

        (data['OFFENSE_CODE_GROUP'] == 'Medical Assistance') |

        (data['OFFENSE_CODE_GROUP'] == 'Investigate Person') |

        (data['OFFENSE_CODE_GROUP'] == 'Others')]

data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'])

topweeks = topweeks.pivot_table(values = 'INCIDENT_NUMBER', index = 'OFFENSE_CODE_GROUP', columns = 'DAY_OF_WEEK', aggfunc = np.size)

topweeks

sns.heatmap(topweeks)

plt.xlabel('Day of the Week')

plt.ylabel('Type of Crime')

plt.title('BOSTON: Heat Map of Day of the Week wise Crimes')

plt.show()

# 6.10.b. Line Graph

plt.figure(figsize=(128,32))

data.groupby(['DAY_OF_WEEK', 'OFFENSE_CODE_GROUP'])['INCIDENT_NUMBER'].count().unstack().plot(marker = 'o')

plt.ylabel('Number of Crimes')

plt.xlabel('Day of the Week')

plt.xticks(np.arange(24), 'DAY_OF_WEEK')

plt.title('BOSTON: Day of the week Crimes During 2015 to 2018')

plt.show()

# Observations:

# 1. Maximum number of Crimes in Boston were observed during the years 2016 & 2017.

# 2. Crime rates have dropped during Winter.

# 3. The trend has remained the same. Almost same no. of cases/crimes have been reported each month in the year from 2015 to 2018.

#    with few exceptions.     

# 4. Motor Vehicle Accident Response tops in the list.

#   Here we have the Top 5 types of Crimes in Boston.

#   a. Motor Vehicle Accident Response tops in the list.

#   b. Larceny

#   c. Medical Assistance

#   d. Investigate Person

#   e. Other

# 5. Maxium number of Crimes occoured on 'Washington Street'.

# 6. Intense amount of criminal activities have happened During Feb during 2016, 2017 & 2018 years.

# 7. Crime rates drop from 1 AM to 6 AM.