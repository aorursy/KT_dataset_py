# libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# load data

print('Loading data...')

train = pd.read_csv('../input/train.csv', na_values=-1).drop(['target'], axis = 1)

test = pd.read_csv('../input/test.csv', na_values=-1)

df = pd.concat([train,test])
f, ax = plt.subplots(1,figsize = (15,5))

sns.boxplot(x="ps_car_11_cat", y="ps_car_06_cat", data=df, ax = ax )

plt.xticks(rotation=90);
df.groupby('ps_car_11_cat')['ps_car_06_cat'].nunique().tail(10)
df[df['ps_car_11_cat']!=104].ps_car_06_cat.value_counts().sort_index()
df[df['ps_car_11_cat']==104].ps_car_06_cat.value_counts().sort_index()
f, ax = plt.subplots(1,figsize = (15,5))

sns.boxplot(x="ps_car_11_cat", y="ps_car_12", data=df, ax = ax )

plt.xticks(rotation=90);
f, ax = plt.subplots(1,figsize = (15,5))

sns.boxplot(x="ps_car_11_cat", y="ps_car_14", data=df, ax = ax )

plt.xticks(rotation=90);
f, ax = plt.subplots(1,figsize = (15,5))

sns.boxplot(x="ps_car_11_cat", y="ps_car_15", data=df, ax = ax )

plt.xticks(rotation=90);