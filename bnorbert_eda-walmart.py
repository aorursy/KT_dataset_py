# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns; sns.set(style="ticks", color_codes=True)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
PATH = '../input/'

FEATURES_PATH = PATH + 'features.csv'

STORES_PATH = PATH + 'stores.csv'

TRAIN_PATH = PATH + 'train.csv'

TEST_PATH = PATH + 'test.csv'
train_df = pd.read_csv(TRAIN_PATH)

features_df = pd.read_csv(FEATURES_PATH)

stores_df = pd.read_csv(STORES_PATH)

test_df = pd.read_csv(TEST_PATH)
test_df.head()
one_df_to_rule_them_all = train_df.merge(stores_df, how='left').merge(features_df, how='left')
one_df_to_rule_them_all.head()
one_ruler = one_df_to_rule_them_all
ans = one_ruler['Store'].unique().shape[0] * one_ruler['Dept'].unique().shape[0]

print("Number of possible store-department pairs: %d" % ans)
print((one_ruler.groupby(['Store', 'Dept']).size()).shape)
cnt = 0

for name, group in one_ruler.groupby(["Store", "Dept"]):

    plt.title(name)

    print(group.shape)

    group = group.sort_values(by=['Date'])

    plt.scatter(range(len(group)), group["Weekly_Sales"])

    plt.show()

    if cnt > 10:

        break

    cnt += 1
def scatter_(df, column):

    plt.figure()

    plt.scatter(df[column] , df['Weekly_Sales'])

    plt.ylabel('Weekly_Sales')

    plt.xlabel(column)
scatter_(one_ruler, 'Fuel_Price')

scatter_(one_ruler, 'Size')

scatter_(one_ruler, 'CPI')

scatter_(one_ruler, 'Type')

scatter_(one_ruler, 'IsHoliday')

scatter_(one_ruler, 'Unemployment')

scatter_(one_ruler, 'Temperature')

scatter_(one_ruler, 'Store')

scatter_(one_ruler, 'Dept')
fig = plt.figure(figsize=(14, 10))

corr = one_ruler.corr()

c = plt.pcolor(corr)

plt.yticks(np.arange(0.5, len(corr.index), 1), corr.index)

plt.xticks(np.arange(0.5, len(corr.columns), 1), corr.columns)

fig.colorbar(c)
sns.pairplot(one_ruler, vars=['Weekly_Sales', 'Fuel_Price', 'Size', 'CPI', 'Dept', 'Temperature', 'Unemployment'])
print(one_ruler.shape)

print(one_ruler[one_ruler['Weekly_Sales'] >= 100000].shape)
df_to_plot = one_ruler[one_ruler['Weekly_Sales'] <= 100000]
sns.jointplot( 'Weekly_Sales', 'Fuel_Price', data=df_to_plot, kind='hex', gridsize=30)

sns.jointplot( 'Weekly_Sales', 'Size', data=df_to_plot, kind='hex', gridsize=30)

sns.jointplot( 'Weekly_Sales', 'CPI', data=df_to_plot, kind='hex', gridsize=30)

sns.jointplot( 'Weekly_Sales', 'Dept', data=df_to_plot, kind='hex', gridsize=30)

sns.jointplot( 'Weekly_Sales', 'Temperature', data=df_to_plot, kind='hex', gridsize=30)

sns.jointplot( 'Weekly_Sales', 'Unemployment', data=df_to_plot, kind='hex', gridsize=30)
sns.countplot(one_ruler['Type'])
print("Min Temperature: ")

print(one_ruler['Temperature'].min())

print("Max Temperature: ")

print(one_ruler['Temperature'].max())

print("Mean Temperature: ")

print(one_ruler['Temperature'].mean())

print("Std Temperature: ")

print(one_ruler['Temperature'].std())
sns.distplot(one_ruler['Temperature'])
df = one_ruler

df['dt'] = pd.to_datetime(one_ruler['Date'])

df_to_plot = df.groupby(['dt'])['Temperature'].mean()



fig, ax = plt.subplots(1, 1, figsize=(8,6))

sns.lineplot(data=df_to_plot, ax=ax)
sns.countplot(one_ruler['IsHoliday'])
sns.distplot(one_ruler['CPI'])
df1 = one_ruler[one_ruler.CPI < 160]

df2 = one_ruler[one_ruler.CPI >= 160]



fig, ax = plt.subplots(2, 1, figsize=(8,14))

sns.countplot(df1['Store'], ax=ax[0])

sns.countplot(df2['Store'], ax=ax[1])
s1 = set(df1['Store'].unique())

s2 = set(df2['Store'].unique())

print(len(s1.intersection(s2)))
fig, ax = plt.subplots(2, 1, figsize=(14,14))



sns.distplot(df1['Temperature'],bins=10, ax=ax[0])

sns.distplot(df2['Temperature'], bins=10, ax=ax[1])

df = df1

df['dt'] = pd.to_datetime(df['Date'])

df_to_plot = df.groupby(['dt'])['CPI'].mean()



fig, ax = plt.subplots(1, 1, figsize=(8,6))

sns.lineplot(data=df_to_plot, ax=ax)
df = df2

df['dt'] = pd.to_datetime(df['Date'])

df_to_plot = df.groupby(['dt'])['CPI'].mean()



fig, ax = plt.subplots(1, 1, figsize=(8,6))

sns.lineplot(data=df_to_plot, ax=ax)
df = one_ruler

df['dt'] = pd.to_datetime(one_ruler['Date'])

df_to_plot = df.groupby(['dt'])['CPI'].mean()



fig, ax = plt.subplots(1, 1, figsize=(8,6))

sns.lineplot(data=df_to_plot, ax=ax)
PATH = '../input/'

FEATURES_PATH = PATH + 'features.csv'

STORES_PATH = PATH + 'stores.csv'

TRAIN_PATH = PATH + 'train.csv'

TEST_PATH = PATH + 'test.csv'

train_df = pd.read_csv(TRAIN_PATH)

features_df = pd.read_csv(FEATURES_PATH)

stores_df = pd.read_csv(STORES_PATH)

test_df = pd.read_csv(TEST_PATH)

one_df_to_rule_them_all = train_df.merge(stores_df, how='left').merge(features_df, how='left')

train_df_merged = one_df_to_rule_them_all

train_df_merged = pd.get_dummies(train_df_merged, columns=['Type'])

train_df_merged.head()
def extract_date_features(df):

    df['Date_dt'] = pd.to_datetime(df['Date'])

    df['day_of_week'] = df['Date_dt'].dt.dayofweek

    df['day'] = df['Date_dt'].dt.day

    df['year'] = df['Date_dt'].dt.year

    df['month'] = df['Date_dt'].dt.month

    df['weight'] = df['IsHoliday'].replace(True,5).replace(False,1)
extract_date_features(train_df_merged)

train_df_merged = train_df_merged.drop(['Date', 'Date_dt', 'IsHoliday'], axis=1)

train_df_merged = train_df_merged.fillna(0)
lst = []

for name, group in train_df_merged.groupby(['Store', 'Dept']):

    lst.append(name)

print(len(lst))
test_df = pd.read_csv(TEST_PATH)

test_df = test_df.merge(stores_df, how='left').merge(features_df, how='left')
train_df_merged.info()
print(test_df.Date.min())

print(test_df.Date.max())
test_df[test_df.Date >= '2013-04-20'].info()
test_df['dt'] = pd.to_datetime(test_df['Date'])
fig, ax = plt.subplots(1, 1, figsize=(12,8))

sns.lineplot(x='dt',y='Unemployment',data=test_df, ax=ax)
fig, ax = plt.subplots(1, 1, figsize=(12,8))

sns.lineplot(x='dt',y='CPI',data=test_df, ax=ax)
one_df_to_rule_them_all['dt'] = pd.to_datetime(one_df_to_rule_them_all['Date'])
fig, ax = plt.subplots(1, 1, figsize=(12,8))

sns.lineplot(x='dt',y='Unemployment',data=one_df_to_rule_them_all, ax=ax)
fig, ax = plt.subplots(1, 1, figsize=(12,8))

sns.lineplot(x='dt',y='CPI',data=one_df_to_rule_them_all, ax=ax)
dfs1_train = one_df_to_rule_them_all[one_df_to_rule_them_all.Store == 1]

dfs1_test = test_df[test_df.Store == 1]
fig, ax = plt.subplots(2, 1, figsize=(12,14))



sns.lineplot(x='dt',y='Unemployment',data=dfs1_train, ax=ax[0])

sns.lineplot(x='dt',y='Unemployment',data=dfs1_test, ax=ax[1])
fig, ax = plt.subplots(2, 1, figsize=(12,14))



sns.lineplot(x='dt',y='CPI',data=dfs1_train, ax=ax[0])

sns.lineplot(x='dt',y='CPI',data=dfs1_test, ax=ax[1])