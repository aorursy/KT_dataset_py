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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_excel('/kaggle/input/abcccc/data.xlsx')
df.head()
#To find unique values from acolumns
def uni(x):
    return x.unique()
df.shape

#a = df.isnull().mean()
plt.figure(figsize=(8,6))
sns.heatmap(df.isnull(),yticklabels=False)
#dropping null values
dff = df.dropna(axis=0)
dff.shape
dff.to_excel('newdata.xlsx', index=False)
# here all the features have very less number of null values. So, not rejecting any feature wrt null values.
plt.figure(figsize=(8,5))
sns.heatmap(dff.isnull(),yticklabels=False)
#df.head(5)
dff['Customer Type'].value_counts()
#working with dff the onw with no null values.
#dfnew.hist(color='green',bins=20,figsize=(20,8))
plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
sns.countplot(x='Customer Type',hue='Class',data=dff,palette='rainbow')
def viv(a,b,c):
    plt.figure(figsize=(8,8))
    sns.set_style('whitegrid')
    sns.countplot(x=a,hue=b,data=c,palette='rainbow')
#to interpret genderwise loyal and disloyal customers
viv(df['Customer Type'],df['Gender'],dff)
#Interpretation:
#1.We can see that the ration of male and female in loyal customer is same and it is also almost same in disloyal customer.
#2. so here gender is not playing in deciding the satisfaction with the flight they are nearly the same.
viv(dff['Gender'],dff['satisfaction'],dff)
#The amount of dis-satisfied males are equal to dis-satisfied females.
#So, we can actually ignore the gender feature.
viv(dff['Customer Type'],dff['Inflight wifi service'],dff)

viv(dff['Customer Type'],dff['Inflight entertainment'],dff)
viv(dff['Customer Type'],dff['On-board service'],dff)
viv(dff['Customer Type'],dff['Leg room service'],dff)

viv(dff['Customer Type'],dff['Baggage handling'],dff)
viv(dff['Customer Type'],dff['Checkin service'],dff)
#On-board service                     
#Leg room service                     
#Baggage handling
#Checkin service 
#Inflight service
#Cleanliness
viv(dff['Customer Type'],dff['Inflight service'],dff)
#On-board service                     0.000023
#Leg room service                     0.000015
#Baggage handling                     0.000015
#Checkin service                      0.000023
#Inflight service                     0.000008
#Cleanliness
viv(dff['Customer Type'],dff['Cleanliness'],dff)
dff['Service'] = ( dff['Cleanliness'] + dff['Inflight service'] + dff['Checkin service'] + dff['Baggage handling'] + 
                    dff['Leg room service'] + dff['On-board service'] + dff['Inflight entertainment'] + dff['Inflight wifi service']) / 8
dff['Service'] = dff['Service'].round()
viv(dff['Customer Type'],dff['Service'],dff)
dff['Flight Distance'].hist(color='green',bins=40,figsize=(8,4))
#from here we can see that airplanes majorly covers less distances than more distance,
#this might be a reason for customers dis-satisfcation that it is not covering more distance
#To check the prefer distances of the customers
sns.barplot(x=dff['Customer Type'], y=dff['Flight Distance'],data=dff)
#We can see that disloyal customers are for less flight distance 
#this can be one of the reasons that airline is not routing for longer distances
sns.barplot(x=dff['Class'], y=dff['Flight Distance'],data=dff)
plt.figure(figsize=(8,8))
sns.barplot(y=dff['Departure Delay in Minutes'],x=dff['Customer Type'],data=dff)
