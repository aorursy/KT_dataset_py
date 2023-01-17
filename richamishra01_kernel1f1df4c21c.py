# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

#/kaggle/input/coronavirusdataset/time.csv

#/kaggle/input/coronavirusdataset/patient.csv

#/kaggle/input/coronavirusdataset/route.csv
patient = pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')

route = pd.read_csv('/kaggle/input/coronavirusdataset/route.csv')

time = pd.read_csv('/kaggle/input/coronavirusdataset/time.csv')
time.head(5)
route.head(5)
del route

del time
patient.info()
patient.isnull().sum()
patient.head(10)
patient.tail(10)
sns.countplot(y=patient['region'],data=patient)
sns.countplot(y=patient['infection_reason'],data=patient)
most_infected_region = patient.loc[(patient['region'] == 'capital area') | (patient['region'] == 'Gyeongsangbuk-do')]
sns.countplot(y= most_infected_region['infection_reason'], hue=most_infected_region['region'])
sns.countplot(x= most_infected_region['state'])
deceased_patient = patient.loc[patient['state'] == 'deceased']

print('Death rate among infected persons are {}'.format(len(deceased_patient)/len(patient)*100))
patient['country'].unique()
china_patient = patient[patient['country'] == 'China']

korea_patient = patient[patient['country'] == 'Korea']

mongolia_patient = patient[patient['country'] == 'Mongolia']

china_patient.isnull().sum()
korea_patient.isnull().sum()
mongolia_patient.isnull().sum()
from datetime import date 

def calculateAge(birthDate): 

    today = date.today() 

    age = today.year - birthDate       

    return age 



#china_patient['age'] = china_patient.apply(lambda x: calculateAge(china_patient['birth_year']))

china_patient['age'] = ((2020 - china_patient['birth_year'])).astype(np.int64) 
sns.distplot(china_patient['age'],hist=True,norm_hist=False)
sns.countplot(x=china_patient['sex'])
len(korea_patient)
sns.countplot(x=korea_patient['sex'])
sns.countplot(x=korea_patient['state'],hue=korea_patient['sex'])
korea_patient['age'] = ((2020 - korea_patient['birth_year']))

sns.distplot(korea_patient['age'].loc[korea_patient['sex'] == 'male'],hist=True,norm_hist=False).set_title('Distribution of male patient in Korea')
sns.distplot(korea_patient['age'].loc[korea_patient['sex'] == 'female'],hist=True,norm_hist=False).set_title('Distribution of female patient in Korea')
sns.countplot(y=korea_patient['infection_reason'],data=korea_patient, hue=korea_patient['sex'])
sns.set(rc={'figure.figsize':(4,6)})

sns.countplot(y=korea_patient['confirmed_date'],data=korea_patient)