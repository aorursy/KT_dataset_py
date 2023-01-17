# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv')
dataset.head()
#To check the null objects and to get info about the type of each column

dataset.info()
#total number of rows and columns

dataset.shape
#To verify that there are no null values in the dataset

dataset.isnull().sum()
#total unique dates present in the dataset

total=dataset['Date'].nunique()

total
#Different countries infected

dataset['Country'].value_counts()
#Date Based Dataset

data_by_date=dataset.groupby('Date').sum()

data_by_date.head()
#Plot showing the comparision (PS: This can be implemented using for loop)



fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(10,7))

axes[0].plot(data_by_date['Cumulative number of case(s)'],data_by_date['Number of deaths'])

axes[1].plot(data_by_date['Number of deaths'],data_by_date['Number recovered'])

axes[2].plot(data_by_date['Number recovered'],data_by_date['Cumulative number of case(s)'])

plt.tight_layout()



axes[0].set_xlabel('Cumulative number of case(s)')

axes[0].set_ylabel('Number of deaths')

axes[1].set_xlabel('Number of deaths')

axes[1].set_ylabel('Number recovered')

axes[2].set_xlabel('Number recovered')

axes[2].set_ylabel('Cumulative number of case(s)')
sns.lmplot(x='Cumulative number of case(s)',y='Number recovered',data=data_by_date)
sns.lmplot(x='Number recovered',y='Number of deaths',data=data_by_date)