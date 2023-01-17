# Test change

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



#list input files available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#read training data

train_df = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])

print("\n---\nnumber of rows x cols in train data: ", train_df.shape, "\n---\n")

#show first 3 rows 

train_df.head(10)
plt.figure(figsize=(12,8))

#distribution of logerror values in 20 bins

sns.distplot(train_df.logerror.values, bins=50, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.ylabel('amount', fontsize=12)

plt.show()

#Insight: in most training cases, Zestimates underestimates the actual price
#Plot logerror field.

#plt.figure(figsize=(8,6))

#scatter plot

#plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))

#sns.distplot(train_df.logerror.values, bins=50, kde=False)

#plt.xlabel('instance_id', fontsize=12)

#plt.ylabel('logerror', fontsize=12)

#plt.show()

#Displays different values for parcelid

(train_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts()



train_df['transaction_month'] = train_df['transactiondate'].dt.month



cnt_srs = train_df['transaction_month'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Month of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()

#read properties data

properties2016 = pd.read_csv("../input/properties_2016.csv")

print("\n---\nnumber of rows x cols in train data: ", properties2016.shape, "\n---\n")

#show first 10 rows 

properties2016.head(10)

properties2016.rows()

properties2016.cols()
