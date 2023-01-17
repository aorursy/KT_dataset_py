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

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sb
df = pd.read_csv('/kaggle/input/mobile-data-speeds-of-all-india-during-march-2018/march18_myspeed.csv')

df.head()
# lets confirm the exact data types of the dataset.



df.dtypes
# we can make Signal_strenght column a numeric type instead

df['Signal_strength'] = pd.to_numeric(df['Signal_strength'], errors='coerce')

df.dtypes
# Confirm the available NaNs and Nulls

df.isna().sum()
#Data Cleanup. Filling the NaNs

#Lets first determine the Skew of the Signal_strength Distribution



plt.hist(df.Signal_strength)

plt.show()
#Because the above Histogram for Signal_strenght is Right skewed, we shall fill our NaNs with the Median



#Define a funcation impute median

def impute_median(series):

    return series.fillna(series.median())
df.Signal_strength = df['Signal_strength'].transform(impute_median)

df.isna().sum()
df.head()
Result1 = df.groupby('Service Provider')['Data Speed(Mbps)'].describe()

Result1
Result1_Mean = df.groupby('Service Provider')['Data Speed(Mbps)'].mean().sort_values(ascending=False)

Result1_Mean
plt.figure(figsize=(10,10))

plt.pie(Result1_Mean, labels=Result1_Mean.index, autopct='%1.1f%%',shadow=True, startangle=90)

plt.title('Best Data Speeds per Service Provider')

plt.tight_layout()

plt.show()
Result2 = df.groupby('Service Provider')['Signal_strength'].describe()

Result2
Result2_mean = df.groupby('Service Provider')['Signal_strength'].mean().sort_values(ascending=False)

Result2_mean
print('The service provider with the worst signal strength is :', Result2_mean.idxmin() )
Result3 = df.groupby('Service Provider').count()

Result3
## Service Provider with most users

plt.figure(figsize=(15, 10))

plt.bar(Result3.index, Result3.Signal_strength)

plt.xlabel('Service Providers')

plt.ylabel('Number of Users')

plt.xticks(Result3.index, rotation=90)

plt.title('Service Provider with most Users')

plt.show()
Result4 = df.groupby('LSA').count()

Result4
## Service Area with most users

plt.figure(figsize=(15, 10))

plt.bar(Result4.index, Result4.Signal_strength)

plt.xlabel('Licensed Service Areas')

plt.ylabel('Number of Users')

plt.xticks(Result4.index, rotation=90)

plt.title('Licensed Service Area with most Users')

plt.show()
# LSA with Speeds Performance

Result5 = df.groupby('LSA')['Data Speed(Mbps)'].describe()

Result5
Result5_mean = df.groupby('LSA')['Data Speed(Mbps)'].mean().sort_values(ascending=False)

Result5_mean
plt.figure(figsize=(15,15))

plt.pie(Result5_mean, labels=Result5_mean.index, autopct='%1.1f%%',shadow=True, startangle=90)

plt.title('Best Data Speeds per LSA')

plt.tight_layout()

plt.show()
# LSA with Signal Strength Performance

Result6 = df.groupby('LSA')['Signal_strength'].describe()

Result6
Result6_mean = df.groupby('LSA')['Signal_strength'].mean().sort_values(ascending=False)

Result6_mean

print('LSA with the worst Signal_strength is:', Result6_mean.idxmin())
df.head()
Result7 = df.groupby('Technology').count()

Result7
#Most Users per Technology



plt.figure(figsize=(5, 10))

plt.bar(Result7.index, Result7.Signal_strength)

plt.xlabel('Technology')

plt.ylabel('Number of Users')

plt.xticks(Result7.index, rotation=90)

plt.title('Technology with most Users')

plt.show()
# Technology with Speeds Performance

Result8 = df.groupby('Technology')['Data Speed(Mbps)'].describe()

Result8
Result8_Mean = df.groupby('Technology')['Data Speed(Mbps)'].mean().sort_values(ascending=False)

Result8_Mean
plt.figure(figsize=(10,10))

plt.pie(Result8_Mean, labels=Result8_Mean.index, autopct='%1.1f%%',shadow=True, startangle=90)

plt.title('Best Data Speeds per Technology')

plt.tight_layout()

plt.show()
# Technology with Signal Strength Performance

Result9 = df.groupby('Technology')['Signal_strength'].describe()

Result9
Result9_Mean = df.groupby('Technology')['Signal_strength'].mean().sort_values(ascending=False)

Result9_Mean
print('Technology with the worst Signal_strength is:', Result9_Mean.idxmin())
#sb.distplot(Result9_Mean)
#plt.figure(figsize=(100,10))

#plt.title()

Result10 = df.groupby('Service Provider')['LSA'].value_counts().unstack(0)



Result10.plot.bar(width=0.8, figsize=(50,20), title='Most Users per Service Provider per LSA')
Result11 = df.groupby('Service Provider')['Technology'].value_counts().unstack(1)

Result11.plot.bar(width=0.8, figsize=(10,10), title='Most Users per Service Provider per Technology')
Result12 = df.groupby('LSA')['Technology'].value_counts().unstack(1)

Result12.plot.bar(width=0.8, figsize=(30,10), title='Most Users per LSA per Technology')