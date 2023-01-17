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
#Import the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import datetime

from dateutil.relativedelta import relativedelta

from datetime import date
PLOT_COLORS = ['#268BD2', "#0052CC", "#FF5722", "#b58900", "#003f5c"]

pd.options.display.float_format = '{:.2f}'.format

sns.set(style = 'ticks')

plt.rc('figure', figsize = (8,5), dpi = 100)

plt.rc('axes', labelpad = 20, facecolor = "#ffffff", linewidth = 0.4, grid = True, labelsize = 14)

plt.rc('patch', linewidth = 0)

plt.rc('xtick.major', width = 0.2)

plt.rc('ytick.major', width = 0.2)

plt.rc('grid', color ='#9E9E9E', linewidth = 0.4)

plt.rc('font', family = 'Arial', weight = '400', size = 10)

plt.rc('text', color = '#282828')

plt.rc('savefig', pad_inches = 0.3, dpi = 300)
path = '/kaggle/input/coronavirusdataset/PatientInfo.csv'

df = pd.read_csv(path)

df.head(3)
#Drop the columns we won't need

df.drop(['patient_id','global_num', 'city', 'birth_year'], axis = 1, inplace = True)
df.isnull().sum()
#Dropping columns having two many missing values

df.drop(['infection_order','infected_by','contact_number','symptom_onset_date'], axis = 1, inplace = True)
#locating data

df1 = df.loc[(df['state'] == 'deceased') & (df['deceased_date'].isnull())]

df1
#dropping those rows located in df1

df.drop([108,284,680,704,706,2554,3073], inplace = True)
#locating data

df2 = df.loc[(df['state'] == 'released') & (df['released_date'].isnull())]

df2
#dropping rows located in df2

df.drop(df2.index, inplace = True)
df.info()
df['released_date'].replace(np.nan, 'None', inplace = True)

df['deceased_date'].replace(np.nan, 'None', inplace = True)
df['disease'].unique()
df['disease'].replace(np.nan, False, inplace = True)
#Dropping missing values

df.dropna(inplace = True)

df.isnull().sum()
#Resetting the index of the dataframe

df.reset_index(drop = True, inplace = True)
df.describe(include = 'all')
sex_count = df['sex'].value_counts().to_dict()



fig,ax = plt.subplots()

plt.pie([sex_count['female'], sex_count['male']], labels = ['FEMALE', 'MALE'], counterclock=False, autopct = '%0.f%%',

      shadow = True, startangle = 90, colors = ['gold', 'green'])

ax.set_title('SEX VALUE COUNT')
df['sex'].value_counts(normalize = True) * 100
province_count = df['province'].value_counts().to_frame().reset_index().rename(columns = {'index': 'province', 'province': 'Value_count'})



fig, ax = plt.subplots()

sns.barplot(x = 'Value_count', y = 'province', data = province_count, palette = sns.cubehelix_palette(n_colors = 20, reverse = True), ax = ax)

ax.set_title('PROVINCE VALUE COUNT')
country_count = df['country'].value_counts().to_frame().reset_index().rename(columns = {'index': 'country', 'country': 'Value_count'})



fig, ax = plt.subplots()

sns.barplot(x = 'Value_count', y = 'country', data = country_count, palette = sns.cubehelix_palette(n_colors = 20, reverse = True), ax = ax)

ax.set_title('COUNTRY VALUE COUNT')
disease_count = df['disease'].value_counts().to_frame().reset_index().rename(columns = {'index': 'disease', 'disease': 'Value_count'})



fig, ax = plt.subplots()

sns.barplot( x = 'disease', y = 'Value_count', data = disease_count, palette = sns.cubehelix_palette(n_colors = 20, reverse = True), ax = ax)

ax.set_title('UNDERLYING DISEASE VALUE COUNT')
infection_count = df['infection_case'].value_counts().to_frame().reset_index().rename(columns = {'index': 'infection case', 'infection_case': 'Value_count'})



fig, ax = plt.subplots()

sns.barplot( x = 'Value_count', y = 'infection case', data = infection_count, palette = sns.cubehelix_palette(n_colors = 20, reverse = True), ax = ax)

ax.set_title('INFECTION CASE VALUE COUNT')
infection_rate = df['infection_case'].value_counts(normalize = True)

infection_rate
state_count = df['state'].value_counts().to_dict()



fig,ax = plt.subplots()

plt.pie([state_count['isolated'], state_count['released'], state_count['deceased']], labels = ['ISOLATED', 'RELEASED', 'DECEASED'], counterclock=False, autopct = '%0.f%%',

      shadow = True, startangle = 90, colors = ['gold', 'green', 'red'])

ax.set_title('STATE VALUE COUNT')
age_count = df['age'].value_counts().to_frame().reset_index().rename(columns = {'index': 'age', 'age': 'Value_count'})



fig, ax = plt.subplots()

sns.barplot( x = 'age', y = 'Value_count', data = age_count, palette = sns.cubehelix_palette(n_colors = 20, reverse = True), ax = ax)

ax.set_title('AGE VALUE COUNT')

#locating data

released_patients = df.loc[df['state'] == 'released']



#converting columns to datetime

released_patients['released_date'] = pd.to_datetime(released_patients['released_date'])

released_patients['confirmed_date'] = pd.to_datetime(released_patients['confirmed_date'])



#Getting the difference between days

released_patients['day_diff'] = released_patients['released_date'] - released_patients['confirmed_date']

released_patients['day_diff'] = released_patients['day_diff']/np.timedelta64(1,'D')



released_patients
released_patients['day_diff'].sum()/ released_patients['day_diff'].count()
released_sex_grouped = released_patients[['sex', 'day_diff']]



sex_grouped = released_sex_grouped.groupby('sex', as_index = False).mean()

sex_grouped
fig, ax = plt.subplots()

sns.barplot( x = 'sex', y = 'day_diff', data = sex_grouped, ax = ax)

ax.set_title('DAY DIFFERENCE BASED ON SEX FOR THE RELEASED PATIENTS')
age_count = released_patients['age'].value_counts().to_frame().reset_index().rename(columns = {'index': 'age', 'age': 'Value_count'})



fig, ax = plt.subplots()

sns.barplot( x = 'age', y = 'Value_count', data = age_count, palette = sns.cubehelix_palette(n_colors = 20, reverse = True), ax = ax)

ax.set_title('AGE VALUE COUNT FOR RELEASED PATIENTS')
#locating data

deceased_patients = df.loc[df['state'] == 'deceased']



deceased_patients['deceased_date'] = pd.to_datetime(deceased_patients['deceased_date'])

deceased_patients['confirmed_date'] = pd.to_datetime(deceased_patients['confirmed_date'])



deceased_patients['day_diff'] = deceased_patients['deceased_date'] - deceased_patients['confirmed_date']

deceased_patients['day_diff'] = deceased_patients['day_diff']/np.timedelta64(1,'D')



deceased_patients
deceased_patients['day_diff'].sum()/ deceased_patients['day_diff'].count()
deceased_sex_grouped = deceased_patients[['sex', 'day_diff']]



sex_grouped = deceased_sex_grouped.groupby('sex', as_index = False).mean()

sex_grouped
fig, ax = plt.subplots()

sns.barplot( x = 'sex', y = 'day_diff', data = sex_grouped, ax = ax)

ax.set_title('DAY DIFFERENCE BASED ON SEX FOR THE DECEASED')
age_count = deceased_patients['age'].value_counts().to_frame().reset_index().rename(columns = {'index': 'age', 'age': 'Value_count'})



fig, ax = plt.subplots()

sns.barplot( x = 'age', y = 'Value_count', data = age_count, palette = sns.cubehelix_palette(n_colors = 20, reverse = True), ax = ax)

ax.set_title('AGE VALUE COUNT FOR DECEASED')
df['month'] = pd.DatetimeIndex(df['confirmed_date']).month



df['month'].replace([1,2,3,4], ['January','February', 'March', 'April'] , inplace = True)

df1 = df['month'].value_counts().to_frame().reset_index().rename(columns = {'index': 'month', 'month': 'Value_count'})



fig, ax = plt.subplots()

sns.lineplot( x = 'month', y = 'Value_count', data = df1)