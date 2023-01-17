import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

import seaborn as sns

from sklearn.preprocessing import StandardScaler



# load the train and test data files

train = pd.read_csv("../input/santander-customer-satisfaction/train.csv", index_col=0)

test = pd.read_csv("../input/santander-customer-satisfaction/test.csv", index_col=0)
print(train.shape)

print(test.shape)
train.head()
train.describe()
train.columns.values
train.dtypes.value_counts()
train.var3.value_counts()
# filter by top countries, excluding the most common one (2)

top_countries = train[(train.var3 != -999999) & (train.var3 != 2)].groupby('var3').filter(lambda x: len(x) > 80)



# plot number of satisfied / unsatisfied customers by country

sns.catplot(x='var3', hue='TARGET', kind='count', data=top_countries);
train.var15.value_counts()
print(train[(train.var15 < 23)].shape)

print(train[(train.var15 < 23)].TARGET.sum())
g = sns.catplot(x='var15', y='TARGET', kind='bar', data=train[(train.var15 > 22) & (train.var15 < 100)], aspect=3)



for ax in g.axes.flat:

    ax.yaxis.set_major_formatter(PercentFormatter())



plt.show();
train.var38.value_counts()
train[(train.var38 != 117310.979016494) & (train.var38 < 300000)].var38.hist(bins=20);
train[(train.var38 != 117310.979016494) & (train.var38 < 300000) & (train.TARGET == 1)].var38.hist(bins=20);
train.TARGET.value_counts(normalize=True) * 100
train.shape
# return a dataset with the columns where any of the values is not 0

train = train.loc[:, (train != 0).any(axis=0)]
train.shape
train.var3 = train.var3.replace(-999999, 2)
train.head()
train.describe()
# standarize our training dataset values and convert it to a new dataframe

# we won't standarize the TARGET feature

train_scaled = StandardScaler().fit_transform(train.drop('TARGET', axis=1).values)

train_scaled_df = pd.DataFrame(train_scaled, index=train.index, columns=train.drop('TARGET', axis=1).columns)

train_scaled_df['TARGET'] = train['TARGET']
train_scaled_df.to_csv('train_clean_standarized.csv')