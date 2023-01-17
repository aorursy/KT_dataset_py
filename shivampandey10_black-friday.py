# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
black = pd.read_csv('../input/BlackFriday.csv')
black.head()
print(black.nunique())
print(black.Gender.value_counts())
gender_nos = black.groupby(['Age','Gender'])['Gender'].count()
gender_nos = gender_nos.unstack(level = 'Gender')
print(gender_nos)
print(black.City_Category.value_counts())
print("Total purchase by male and female is \n",black.groupby('Gender')['Purchase'].sum())
print(black.groupby('City_Category')['Purchase'].sum())
black.groupby('City_Category')['Purchase'].sum().plot(kind = 'bar')
plt.xticks(rotation = 0)
plt.ylabel("Total purchase")
print(black.Stay_In_Current_City_Years.value_counts(normalize = True))
print(black.groupby(['Gender','City_Category'])['Purchase'].sum())
black.groupby(['Gender','City_Category'])['Purchase'].sum().plot(kind = 'barh')
plt.xlabel('Total purchase')
print("Total percentage of male and female in each city category")
print("----Female----")
print(black.City_Category[black.Gender == 'F'].value_counts(normalize = True))
print("----Male----")
print(black.City_Category[black.Gender == 'M'].value_counts(normalize = True))
city = black.groupby(['Age','City_Category'])['Purchase'].sum()
print(city.unstack(level = 'City_Category'))
city.unstack(level = 'City_Category').plot(kind = 'bar')
print(black.Marital_Status.value_counts())
purchase_marital = black.groupby(['Marital_Status','City_Category'])['Purchase'].sum().unstack(level = 
                                                                                'City_Category')
print(purchase_marital)
purchase_marital.plot(kind = 'bar')
plt.xticks(rotation = 0)
hist = black.Age.value_counts()
print(hist)
sns.set()
hist.plot(kind = 'bar')
total_purchase = black.groupby(['Age','Gender','City_Category'])['Purchase'].sum()
print("Total purchase of male and female in age groups from every city category are:\n"
      ,total_purchase.unstack(level = 'City_Category'))
print(black.groupby(['Occupation','City_Category'])['Purchase'].sum().unstack(level = 'City_Category'))
black.groupby(['Occupation','City_Category'])['Purchase'].sum().unstack(level = 'City_Category').plot(kind = 'bar')
print(black.groupby(['Stay_In_Current_City_Years','City_Category'])['City_Category'].count().
      unstack(level = 'City_Category'))
print("we can assume that 0 is for unmarried and 1 is for married:  \n", 
      black.Marital_Status.value_counts())
black.groupby(['Marital_Status','Age'])['Purchase'].sum().plot(kind = 'bar')
plt.ylabel('Total Purchase')
plt.xticks(rotation = 75)
black.groupby(['Marital_Status','Occupation'])['Purchase'].sum().unstack(level='Marital_Status')
black.groupby(['Marital_Status','City_Category','Gender'])['Purchase'].count().unstack(level = 'City_Category')
black.groupby(['Marital_Status','Stay_In_Current_City_Years','City_Category','Gender'])['Purchase'].count().unstack(level ='Stay_In_Current_City_Years')
