# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



data_path = "../input/"

train_file = data_path + "train_ver2.csv"

test_file = data_path + "test_ver2.csv"
train = pd.read_csv(train_file, usecols=['age'])

train.head()
print(list(train.age.unique()))
train['age'] = train['age'].replace(to_replace=[' NA'], value=np.nan)
train['age'] = train['age'].astype('float64')



age_series = train.age.value_counts()

plt.figure(figsize=(12,4))

sns.barplot(age_series.index.astype('int'), age_series.values, alpha=0.8)

plt.ylabel('Number of Occurrences of the customer', fontsize=12)

plt.xlabel('Age', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
train.age.isnull().sum()
train.age.mean()
test = pd.read_csv(test_file, usecols=['age'])

test['age'] = test['age'].replace(to_replace=[' NA'], value=np.nan)

test['age'] = test['age'].astype('float64')



age_series = test.age.value_counts()

plt.figure(figsize=(12,4))

sns.barplot(age_series.index.astype('int'), age_series.values, alpha=0.8)

plt.ylabel('Number of Occurrences of the customer', fontsize=12)

plt.xlabel('Age', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
train = pd.read_csv(train_file, usecols=['antiguedad'])

train.head()
print(list(train.antiguedad.unique()))
train['antiguedad'] = train['antiguedad'].replace(to_replace=['     NA'], value=np.nan)

train.antiguedad.isnull().sum()
train['antiguedad'] = train['antiguedad'].astype('float64')

(train['antiguedad'] == -999999.0).sum()
col_series = train.antiguedad.value_counts()

plt.figure(figsize=(12,4))

sns.barplot(col_series.index.astype('int'), col_series.values, alpha=0.8)

plt.ylabel('Number of Occurrences of the customer', fontsize=12)

plt.xlabel('Customer Seniority', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
test = pd.read_csv(test_file, usecols=['antiguedad'])

test['antiguedad'] = test['antiguedad'].replace(to_replace=[' NA'], value=np.nan)

test['antiguedad'] = test['antiguedad'].astype('float64')



col_series = test.antiguedad.value_counts()

plt.figure(figsize=(12,4))

sns.barplot(col_series.index.astype('int'), col_series.values, alpha=0.8)

plt.ylabel('Number of Occurrences of the customer', fontsize=12)

plt.xlabel('Customer Seniority', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
train = pd.read_csv(train_file, usecols=['renta'])

train.head()
unique_values = np.sort(train.renta.unique())

plt.scatter(range(len(unique_values)), unique_values)

plt.show()
train.renta.mean()
train.renta.median()
train.renta.isnull().sum()
train.fillna(101850., inplace=True) #filling NA as median for now

quantile_series = train.renta.quantile(np.arange(0.99,1,0.001))

plt.figure(figsize=(12,4))

sns.barplot((quantile_series.index*100), quantile_series.values, alpha=0.8)

plt.ylabel('Rent value', fontsize=12)

plt.xlabel('Quantile value', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
rent_max_cap = train.renta.quantile(0.999)

train['renta'][train['renta']>rent_max_cap] = 101850.0 # assigining median value 

sns.boxplot(train.renta.values)

plt.show()
test = pd.read_csv(test_file, usecols=['renta'])

test['renta'] = test['renta'].replace(to_replace=['         NA'], value=np.nan).astype('float') # note that there is NA value in test

unique_values = np.sort(test.renta.unique())

plt.scatter(range(len(unique_values)), unique_values)

plt.show()
test.renta.mean()
test.fillna(101850., inplace=True) #filling NA as median for now

quantile_series = test.renta.quantile(np.arange(0.99,1,0.001))

plt.figure(figsize=(12,4))

sns.barplot((quantile_series.index*100), quantile_series.values, alpha=0.8)

plt.ylabel('Rent value', fontsize=12)

plt.xlabel('Quantile value', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
test['renta'][test['renta']>rent_max_cap] = 101850.0 # assigining median value 

sns.boxplot(test.renta.values)

plt.show()
train = pd.read_csv(data_path+"train_ver2.csv", nrows=100000)

target_cols = ['ind_cco_fin_ult1', 'ind_cder_fin_ult1',

                             'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',

                             'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',

                             'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',

                             'ind_deme_fin_ult1', 'ind_dela_fin_ult1',

                             'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',

                             'ind_hip_fin_ult1', 'ind_plan_fin_ult1',

                             'ind_pres_fin_ult1', 'ind_reca_fin_ult1',

                             'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',

                             'ind_viv_fin_ult1', 'ind_nomina_ult1',

                             'ind_nom_pens_ult1', 'ind_recibo_ult1']

train[target_cols] = (train[target_cols].fillna(0))

train["age"] = train['age'].map(str.strip).replace(['NA'], value=0).astype('float')

train["antiguedad"] = train["antiguedad"].map(str.strip)

train["antiguedad"] = train['antiguedad'].replace(['NA'], value=0).astype('float')

train["antiguedad"].ix[train["antiguedad"]>65] = 65 # there is one very high skewing the graph

train["renta"].ix[train["renta"]>1e6] = 1e6 # capping the higher values for better visualisation

train.fillna(-1, inplace=True)
fig = plt.figure(figsize=(16, 120))

numeric_cols = ['age', 'antiguedad', 'renta']

#for ind1, numeric_col in enumerate(numeric_cols):

plot_count = 0

for ind, target_col in enumerate(target_cols):

    for numeric_col in numeric_cols:

        plot_count += 1

        plt.subplot(22, 3, plot_count)

        sns.boxplot(x=target_col, y=numeric_col, data=train)

        plt.title(numeric_col+" Vs "+target_col)

plt.show()