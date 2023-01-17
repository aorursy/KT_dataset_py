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
df = pd.read_csv('/kaggle/input/gender-equality-un-unpaid-work-data/unpaid_work.csv',encoding = "ISO-8859-1")
df.shape #796,8

#check for null values

df.isna().sum() #Area field has most of the null values, thus can be avoided being used in analysis
df.dtypes
df_copy = df.copy()

df_copy['Age'].value_counts()

#Age colummn is differently classified and can be standardised based on EOCD labour classification

#15-24

#25-54

#55-64
df_copy['Survey Availability'].value_counts()

#Upto 1984

#1985-1994

#1995-2004

#1997-1998 

#1998-00 

#2004-05

#2006-07

#2005 and later

#2015

#Replace and change the groups for 10 years data each

df_copy['Survey Availability'] = df_copy['Survey Availability'].map({'Up to 1984':'Up to 1984','1985 - 1994':'1985 - 1994',

                                                                     '1995 - 2004':'1995 - 2004','2005 and later':'2005 and later',

                                                                     '1997-98': '1995 - 2004', '1998-00': '1995 - 2004', 

                                                                     '2004-05':'2005 and later', '2006-07':'2005 and later'})



df_copy['Survey Availability'].value_counts()

#after changing

#upto 1984

#1985-1994

#1995-2004

#2005 and later

#Now it's much better
df_copy['Time use'].value_counts()
#group data by survey availability,country, gender,time use and age

df_copy.shape #796, 8

df_new = df_copy.groupby(['Survey Availability','Year','Country','Gender','Time use','Age']).mean().reset_index()
%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import numpy as np

fig = plt.figure()

ax = plt.axes()

plt.plot(df_new['Average Time (hours)'], color='g')  # solid green
df_new.shape #752, 7

#We are just left with age column to be cleaned

df_new.Age.value_counts()
#we will change the age column to have three different columns, lower, upper and average age value

df_new['Age'] = df_new['Age'].str.replace('+',' to 84',regex=True) #Adding end range as 84 as this is the standard maximum value 

                                                                   #considered in maxiumum cases



df_new['min_age'] = df_new.Age.str.split('\s+').str[0]

df_new['max_age'] = df_new.Age.str.split('\s+').str[-1]



df_new.head()
df_new['min_age'].value_counts()

df_new['max_age'].loc[(df_new['max_age'] == "age")] = "Unknown"

df_new['max_age'] = df_new['max_age'].str.replace('?','',regex=True)

df_new['max_age'].value_counts()



df_unknown = df_new[(df_new['min_age']=='Unknown') & (df_new['max_age']=='Unknown')]

df_known = df_new[(df_new['min_age']!='Unknown') & (df_new['max_age'] != 'Unknown')]
df_known['min_age']=df_known['min_age'].astype(int)

df_known['max_age']=df_known['max_age'].astype(int)

df_known['avg_age'] = (df_known['min_age']+df_known['max_age'])//2

df_unknown['avg_age'] = "Unknown"
df_known.head()

df_unknown.head()

#merge both df

df_new = df_known.append(df_unknown)

df_new.shape
#Create a time use flag unpaid and paid for better visualization

df_new['timeuseflag'] = df_new['Time use'].apply(lambda x: 'Paid' if 'Unpaid' not in x else 'Unpaid')



df_new.head()

#df['new column name'] = df['column name'].apply(lambda x: 'value if condition is met' if x condition else 'value if condition is not met')
df_new.to_csv('unpaid_work_final.csv')