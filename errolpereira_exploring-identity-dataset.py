import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import altair as alt

import gc #garbage collection 

gc.enable()  #enabling garbage collection



#suppressing warnings.

import warnings

warnings.filterwarnings('ignore')
#Data path

PATH = '/kaggle/input/ieee-fraud-detection/'
#Reading the train and test identity dataset.

train_identity = pd.read_csv(f'{PATH}train_identity.csv')

test_identity = pd.read_csv(f'{PATH}test_identity.csv')
#shape of the dataframe

print(f'Train Shape: {train_identity.shape}')

print(f'Test Shape: {test_identity.shape}')
#exploring first five observations.

train_identity.head()
#Exploring id_01.

train_identity.id_01.describe()
train_identity.id_01.value_counts(dropna=False)
#id_02.

train_identity.id_02.describe()
train_identity.id_02.value_counts(dropna=False)
#distribution of the data.

sns.distplot(train_identity.id_02, bins=20);
#3. id_03.

train_identity.id_03.describe()
train_identity.id_03.value_counts(dropna=False)
#4.id_04.

train_identity.id_04.describe()
train_identity.id_04.value_counts(dropna=False)
#id_05.

train_identity.id_05.describe()
train_identity.id_05.value_counts(dropna=False)
#id_06.

train_identity.id_06.describe()
train_identity.id_06.value_counts(dropna=False)
#id_07.

train_identity.id_07.describe()
train_identity.id_07.value_counts(dropna=False)
#id_08

train_identity.id_08.describe()
train_identity.id_08.value_counts(dropna=False)
#id_09

train_identity.id_09.describe()
train_identity.id_09.value_counts(dropna=False)
#id_10

train_identity.id_10.describe()
train_identity.id_10.value_counts(dropna=False)
#id_11

train_identity.id_11.describe()
train_identity.id_11.value_counts(dropna=False)
#id_12

train_identity.id_12.describe()
#count plot.

sns.countplot(train_identity.id_12)

print(f'NAN count: {train_identity.id_12.isnull().sum()}')
#id_13

train_identity.id_13.describe()
train_identity.id_13.value_counts(dropna=False)
#id_14

train_identity.id_14.describe()
train_identity.id_14.value_counts(dropna=False)
#id_15

train_identity.id_15.describe()
print(f'NAN count: {train_identity.id_15.isnull().sum()}')

sns.countplot(train_identity.id_15);
#id_16

train_identity.id_16.describe()
print(f'NAN count: {train_identity.id_16.isnull().sum()}')

sns.countplot(train_identity.id_16);
#id_17

train_identity.id_17.describe()
train_identity.id_17.value_counts(dropna=False)
#id_18.

train_identity.id_18.describe()
#id_19.

train_identity.id_19.describe()
#id_20.

train_identity.id_20.describe()
#id_21.

train_identity.id_21.describe()
#id_22.

train_identity.id_22.describe()
#id_23.

train_identity.id_23.describe()
print(f'NAN count: {train_identity.id_23.isnull().sum()}')

sns.countplot(train_identity.id_23)

plt.xticks(rotation=90);
#id_24.

train_identity.id_24.describe()
#id_25.

train_identity.id_25.describe()
#id_26.

train_identity.id_26.describe()
#id_27.

train_identity.id_27.describe()
#id_28.

train_identity.id_28.describe()
#id_29.

train_identity.id_29.describe()
#id_30.

train_identity.id_30.describe()
print(f'NAN count: {train_identity.id_30.isnull().sum()}')

plt.figure(figsize=(13, 8))

sns.countplot(train_identity.id_30)

plt.xticks(rotation=90);
#id_31.

train_identity.id_31.describe()
print(f'NAN count: {train_identity.id_31.isnull().sum()}')

plt.figure(figsize=(19, 8))

sns.countplot(train_identity.id_31)

plt.xticks(rotation=90);
#id_32.

train_identity.id_32.describe()
#id_33.

train_identity.id_33.describe()
print(f'NAN count: {train_identity.id_33.isnull().sum()}')

plt.figure(figsize=(19, 8))

sns.countplot(train_identity.id_33.iloc[:30000])

plt.xticks(rotation=90);
#id_34.

train_identity.id_34.describe()
print(f'NAN count: {train_identity.id_34.isnull().sum()}')

plt.figure(figsize=(9, 5))

sns.countplot(train_identity.id_34.iloc[:30000])

plt.xticks(rotation=90);
#id_35.

train_identity.id_35.describe()
print(f'NAN count: {train_identity.id_35.isnull().sum()}')

plt.figure(figsize=(9, 5))

sns.countplot(train_identity.id_35)

plt.xticks(rotation=90);
#id_36.

train_identity.id_36.describe()
print(f'NAN count: {train_identity.id_36.isnull().sum()}')

plt.figure(figsize=(9, 5))

sns.countplot(train_identity.id_36)

plt.xticks(rotation=90);
#id_37.

train_identity.id_37.describe()
print(f'NAN count: {train_identity.id_37.isnull().sum()}')

plt.figure(figsize=(9, 5))

sns.countplot(train_identity.id_37)

plt.xticks(rotation=90);
#id_38

train_identity.id_38.describe()
print(f'NAN count: {train_identity.id_38.isnull().sum()}')

plt.figure(figsize=(9, 5))

sns.countplot(train_identity.id_38)

plt.xticks(rotation=90);
#Device Info

train_identity.DeviceInfo.describe()
print(f'NAN count: {train_identity.DeviceInfo.isnull().sum()}')

plt.figure(figsize=(19, 5))

sns.countplot(train_identity.DeviceInfo[:500])

plt.xticks(rotation=90);
#Device Type

train_identity.DeviceType.describe()
print(f'NAN count: {train_identity.DeviceType.isnull().sum()}')

plt.figure(figsize=(9, 5))

sns.countplot(train_identity.DeviceType)

plt.xticks(rotation=90);
cols = ['id_0'+str(i) for i in range(1, 10)]

cols += ['id_10', 'id_11']



#selecting only numerical columns

train_num = train_identity[cols]

#correlation

corr = train_num.corr()



#heatmap

plt.figure(figsize=(15,9))

sns.heatmap(corr, annot=True, vmax=1., vmin=-1.)
#distribution.

sns.distplot(train_identity.id_03.dropna())
sns.distplot(train_identity.id_09.dropna())
#All variables correlation.

cols = [col for col in train_identity.columns if train_identity[col].dtype != 'O']

cols.remove('TransactionID')



#selecting only numerical columns

train_num = train_identity[cols]

#correlation

corr = train_num.corr()



#heatmap

plt.figure(figsize=(21, 21))

sns.heatmap(corr, annot=True, vmax=1., vmin=-1., cmap='mako')
#checking the categrical variables containing in the train sets are also present in the test set.

def checkcat(df):

    for col in df.columns:

        length = len(set(test_identity[col].values) - set(train_identity[col].values))

        if length > 0:

            print(f'{col} in the test set has {length} values that are not present in the train set')



cat_cols = [col for col in train_identity.columns if train_identity[col].dtype == 'O']

#cat_cols += ['id_13', 'id_14', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22',

#           'id_24', 'id_25', 'id_26', 'id_32']

print(cat_cols)
checkcat(train_identity[cat_cols])