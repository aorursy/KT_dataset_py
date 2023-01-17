# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualisiton

import seaborn as sns #data visualisiton



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Import the dataset

df_1 = pd.read_csv('../input/chennai_reservoir_levels.csv')

df_2 = pd.read_csv('../input/chennai_reservoir_rainfall.csv')
#Lets a qick look of the dataset

df_1.info()

df_2.info()
#Lets take date to index to our dataset

df_1['Date'] = pd.to_datetime(df_1['Date'])

df_1.index = pd.to_datetime(df_1['Date'])

df_1.head()

#Same prosses apply to the rainfall dataset

df_2['Date'] = pd.to_datetime(df_2['Date'])

df_2.index = pd.to_datetime(df_2['Date'])

df_2.head(3)
#Now delete the date column

del df_1['Date']

df_1.head()

del df_2['Date']

df_2.head()


#Now lets see the statistical inferance of the water level dataset

df_1.describe()

#Now lets see the statistical inferance of the rainfall dataset

df_2.describe()
#POONDI

sns.distplot(df_1['POONDI'], color = 'red')

plt.title('Distribution of water levels of POONDI', fontsize = 20)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()
#CHOLAVARAM

sns.distplot(df_1['CHOLAVARAM'], color = 'blue')

plt.title('Distribution of water levels of CHOLAVARAM', fontsize = 20)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()
#REDHILLS

sns.distplot(df_1['REDHILLS'], color = 'green')

plt.title('Distribution of water levels of REDHILLS', fontsize = 20)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()
#CHEMBARAMBAKKAM

sns.distplot(df_1['CHEMBARAMBAKKAM'], color = 'purple')

plt.title('Distribution of water levels of CHEMBARAMBAKKAM', fontsize = 20)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()
#POONDI

sns.distplot(df_2['POONDI'], color = 'red')

plt.title('Distribution of rainfall in POONDI', fontsize = 20)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()
#CHOLAVARAM

sns.distplot(df_1['CHOLAVARAM'], color = 'blue')

plt.title('Distribution of rainfall in CHOLAVARAM', fontsize = 20)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()
#REDHILLS

sns.distplot(df_1['REDHILLS'], color = 'green')

plt.title('Distribution of rainfall in REDHILLS', fontsize = 20)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()
#CHEMBARAMBAKKAM

sns.distplot(df_1['CHEMBARAMBAKKAM'], color = 'purple')

plt.title('Distribution of rainfall in CHEMBARAMBAKKAM', fontsize = 20)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()
#Pairplot of water level

sns.pairplot(df_1)

plt.title('Pairplot of the water lavels', fontsize = 20)

plt.show()
#Pairplot of rainfall

sns.pairplot(df_2)

plt.title('Pairplot of the rainfalls', fontsize = 20)

plt.show()

#POONDI

plt.plot(df_1['POONDI'], color = 'red')

plt.title('POONDI-Water Level')

plt.xlabel('Year')

plt.ylabel('Water Level')

plt.show()
#CHOLAVARAM

plt.plot(df_1['CHOLAVARAM'], color = 'green')

plt.title('CHOLAVARAM-Water Level')

plt.xlabel('Year')

plt.ylabel('Water Level')

plt.show()
#REDHILLS

plt.plot(df_1['REDHILLS'], color = 'blue')

plt.title('REDHILLS-Water Level')

plt.xlabel('Year')

plt.ylabel('Water Level')

plt.show()
#CHEMBARAMBAKKAM

plt.plot(df_1['CHEMBARAMBAKKAM'], color = 'M')

plt.title('CHEMBARAMBAKKAM-Water Level')

plt.xlabel('Year')

plt.ylabel('Water Level')

plt.show()
df_1['TOTAL'] = df_1['POONDI'] + df_1['CHOLAVARAM'] + df_1['REDHILLS'] + df_1['CHEMBARAMBAKKAM']
#Total water level

plt.plot(df_1['TOTAL'], color = 'deeppink')

plt.title('TOTAL-Water Level')

plt.xlabel('Year')

plt.ylabel('Water Level')

plt.show()
#POONDI

plt.plot(df_2['POONDI'], color = 'red')

plt.title('POONDI-Water Level')

plt.xlabel('Year')

plt.ylabel('Water Level')

plt.show()

#CHOLAVARAM

plt.plot(df_2['CHOLAVARAM'], color = 'green')

plt.title('CHOLAVARAM-Water Level')

plt.xlabel('Year')

plt.ylabel('Water Level')

plt.show()
#REDHILLS

plt.plot(df_2['REDHILLS'], color = 'blue')

plt.title('REDHILLS-Water Level')

plt.xlabel('Year')

plt.ylabel('Water Level')

plt.show()
#CHEMBARAMBAKKAM

plt.plot(df_2['CHEMBARAMBAKKAM'], color = 'M')

plt.title('CHEMBARAMBAKKAM-Water Level')

plt.xlabel('Year')

plt.ylabel('Water Level')

plt.show()
df_2['TOTAL'] = df_2['POONDI'] + df_2['CHOLAVARAM'] + df_2['REDHILLS'] + df_2['CHEMBARAMBAKKAM']
#Total rainfall

plt.plot(df_1['TOTAL'], color = 'deeppink')

plt.title('TOTAL-Water Level')

plt.xlabel('Year')

plt.ylabel('Water Level')

plt.show()