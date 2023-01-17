# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # Visualising data

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
renfe = pd.read_csv("../input/renfe.csv")
renfe.info()
renfe.isnull().sum()
renfe.train_class.value_counts()

renfe.fare.value_counts()
renfe.dropna(axis=0,subset=['price'],inplace=True)
renfe.train_class.fillna('Turista',inplace=True)

renfe.fare.fillna('Promo',inplace=True)
renfe = renfe.loc[:, ~renfe.columns.str.contains('Unnamed')]

renfe.drop('insert_date',inplace=True,axis=1)
for i in ['start_date','end_date']:

    renfe[i] = pd.to_datetime(renfe[i])



renfe['time_to_travel'] = renfe['end_date'] - renfe['start_date']

renfe['day_of_week'] = renfe['start_date'].dt.weekday_name

renfe['date'] = renfe['start_date'].dt.date

renfe['time_in_seconds'] = renfe['time_to_travel'].dt.total_seconds()
renfe_greater_150 = renfe[(renfe['price'] > 150)]

renfe_between_100_150 = renfe[(renfe['price'] > 100) & (renfe['price'] < 150)]

renfe_between_50_100 = renfe[(renfe['price'] > 50) & (renfe['price'] < 100)]

renfe_less_50 = renfe[(renfe['price'] < 50)]
sns.jointplot("time_in_seconds","price",data=renfe_greater_150)

sns.catplot(data=renfe_greater_150,x="train_type",y="price")

sns.catplot(data=renfe_greater_150,x="fare",y="price")
sns.catplot(data=renfe_greater_150,x="train_type",y="price")
sns.jointplot("time_in_seconds","price",data=renfe_between_100_150)

sns.catplot(data=renfe_between_100_150,x="train_type",y="price")

sns.catplot(data=renfe_between_100_150,x="fare",y="price")

percentage_100_150 = len(renfe_between_100_150)/ len(renfe) * 100

print(percentage_100_150)
sns.jointplot("time_in_seconds","price",data=renfe_between_50_100)

sns.catplot(data=renfe_between_50_100,x="train_type",y="price")

sns.catplot(data=renfe_between_50_100,x="fare",y="price")

percentage_50_100 = len(renfe_between_50_100)/ len(renfe) * 100

print(percentage_50_100)
sns.jointplot("time_in_seconds","price",data=renfe_less_50)

sns.catplot(data=renfe_less_50,x="train_type",y="price")

sns.catplot(data=renfe_less_50,x="fare",y="price")

percentage_lesser_50 = len(renfe_less_50)/ len(renfe) * 100

print(percentage_lesser_50)
sns.barplot(x=renfe.day_of_week.value_counts(),y=renfe.day_of_week.value_counts().index)