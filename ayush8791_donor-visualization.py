# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
#Reading data into dataframes
donations=pd.read_csv('../input/Donations.csv')
donors=pd.read_csv('../input/Donors.csv')
projects=pd.read_csv('../input/Projects.csv')
resources=pd.read_csv('../input/Resources.csv')
schools=pd.read_csv('../input/Schools.csv')
teachers=pd.read_csv('../input/Teachers.csv')
donations.head()
donors.head()
projects.head()
resources.head()
schools.head()
teachers.head()
total=donations.isnull().sum()
percent=(total/donations.isnull().count()*100)
missing_data_donations=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data_donations.head()
total=donors.isnull().sum().sort_values(ascending=False)
percent=(donors.isnull().sum()/donors.isnull().count()*100).sort_values(ascending=False)
missing_data_donors=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data_donors.head()
total=projects.isnull().sum().sort_values(ascending=False)
percent=(projects.isnull().sum()/projects.isnull().count()*100).sort_values(ascending=False)
missing_data_projects=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data_projects.head()
total=resources.isnull().sum().sort_values(ascending=False)
percent=(resources.isnull().sum()/resources.isnull().count()*100).sort_values(ascending=False)
missing_data_res=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data_res.head()
total=schools.isnull().sum().sort_values(ascending=False)
percent=(schools.isnull().sum()/schools.isnull().count()*100).sort_values(ascending=False)
missing_data_schools=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data_schools.head()
total=teachers.isnull().sum().sort_values(ascending=False)
percent=(teachers.isnull().sum()/teachers.isnull().count()*100).sort_values(ascending=False)
missing_data_teachers=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data_teachers.head()
donations['Donation Amount'].describe()


# Merge donation data with donor data 
donors_donations = donations.merge(donors, on='Donor ID', how='inner')
donors_donations["Donor City"].value_counts()







