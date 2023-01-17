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

import seaborn as sns



#read data into dataframe named 'wage'

import pandas as pd

wage = pd.read_csv("../input/City_of_Seattle_Wage_Data.csv")



# Check if there is missing value in dataset

wage.isnull().sum()



#check the last 5 rows in dataset

wage.tail()
#remove * and leading space 

wage['Job Title'] = wage['Job Title'].str.replace('*','').str.strip()



#remove leading space in column label - 'Hourly Rate '

wage.rename(columns={'Hourly Rate ':'Hourly Rate'}, inplace=True)



#view update

wage.tail()

wage['Department'].value_counts()
f, ax = plt.subplots(figsize=(8, 15))

sns.boxplot(x='Hourly Rate', y='Department', data=wage)

plt.title('Distribution of Hourly Rates')
wage['Hourly Rate'].describe()
hist_plot=wage.hist(column='Hourly Rate', bins=100, figsize=(8,8))

plt.title('Histogram of Hourly Rate among City of Seattle Employees in Jan 2020')

plt.xlabel('Hourly Rate')

plt.ylabel('Frequency')
def underpaid(rate):

    if rate < 16.39:

         return ("Below Min. Wage")

    elif rate >= 16.39:

        return ("Meet Min. Wage")
wage['Hourly Rate'].apply(underpaid).value_counts()
wage[wage['Hourly Rate']<16.39]
wage.nlargest(10, ['Hourly Rate'])
wage[wage['Job Title'].str.contains('CEO')]
wage[wage['Department'].str.contains('Seattle City Light')].boxplot()

wage[wage['Department'].str.contains('Seattle Public Utilities')].boxplot()