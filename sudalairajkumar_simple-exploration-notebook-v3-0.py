# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



color = sns.color_palette()



data_path = "../input/"

train_file = data_path + "train_ver2.csv"

test_file = data_path + "test_ver2.csv"
train = pd.read_csv(data_path+train_file, usecols=['ncodpers'])

test = pd.read_csv(data_path+test_file, usecols=['ncodpers'])

print("Number of rows in train : ", train.shape[0])

print("Number of rows in test : ", test.shape[0])
train_unique_customers = set(train.ncodpers.unique())

test_unique_customers = set(test.ncodpers.unique())

print("Number of customers in train : ", len(train_unique_customers))

print("Number of customers in test : ", len(test_unique_customers))

print("Number of common customers : ", len(train_unique_customers.intersection(test_unique_customers)))
num_occur = train.groupby('ncodpers').agg('size').value_counts()



plt.figure(figsize=(8,4))

sns.barplot(num_occur.index, num_occur.values, alpha=0.8, color=color[0])

plt.xlabel('Number of Occurrences of the customer', fontsize=12)

plt.ylabel('Number of customers', fontsize=12)

plt.show()
del train_unique_customers

del test_unique_customers
train = pd.read_csv(data_path+"train_ver2.csv", dtype='float16', 

                    usecols=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 

                             'ind_cco_fin_ult1', 'ind_cder_fin_ult1',

                             'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',

                             'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',

                             'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',

                             'ind_deme_fin_ult1', 'ind_dela_fin_ult1',

                             'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',

                             'ind_hip_fin_ult1', 'ind_plan_fin_ult1',

                             'ind_pres_fin_ult1', 'ind_reca_fin_ult1',

                             'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',

                             'ind_viv_fin_ult1', 'ind_nomina_ult1',

                             'ind_nom_pens_ult1', 'ind_recibo_ult1'])
target_counts = train.astype('float64').sum(axis=0)

#print(target_counts)

plt.figure(figsize=(8,4))

sns.barplot(target_counts.index, target_counts.values, alpha=0.8, color=color[0])

plt.xlabel('Product Name', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
train = pd.read_csv(data_path+"train_ver2.csv", usecols=['fecha_dato', 'fecha_alta'], parse_dates=['fecha_dato', 'fecha_alta'])

train['fecha_dato_yearmonth'] = train['fecha_dato'].apply(lambda x: (100*x.year) + x.month)

yearmonth = train['fecha_dato_yearmonth'].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(yearmonth.index, yearmonth.values, alpha=0.8, color=color[0])

plt.xlabel('Year and month of observation', fontsize=12)

plt.ylabel('Number of customers', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
train['fecha_alta_yearmonth'] = train['fecha_alta'].apply(lambda x: (100*x.year) + x.month)

yearmonth = train['fecha_alta_yearmonth'].value_counts()

print("Minimum value of fetcha_alta : ", min(yearmonth.index))

print("Maximum value of fetcha_alta : ", max(yearmonth.index))



plt.figure(figsize=(12,4))

sns.barplot(yearmonth.index, yearmonth.values, alpha=0.8, color=color[1])

plt.xlabel('Year and month of joining', fontsize=12)

plt.ylabel('Number of customers', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
year_month = yearmonth.sort_index().reset_index()

year_month = year_month.ix[185:]

year_month.columns = ['yearmonth', 'number_of_customers']



plt.figure(figsize=(12,4))

sns.barplot(year_month.yearmonth.astype('int'), year_month.number_of_customers, alpha=0.8, color=color[2])

plt.xlabel('Year and month of joining', fontsize=12)

plt.ylabel('Number of customers', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
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
cols = ["ind_empleado","pais_residencia","sexo","ind_nuevo","indrel","ult_fec_cli_1t","indrel_1mes","tiprel_1mes","indresi","indext","conyuemp","canal_entrada","indfall","tipodom","cod_prov","nomprov","ind_actividad_cliente","segmento"]

for col in cols:

    train = pd.read_csv("../input/train_ver2.csv", usecols = ["ncodpers", col], nrows=1000000)

    train = train.fillna(-99)

    len_unique = len(train[col].unique())

    print("Number of unique values in ",col," : ",len_unique)

    if len_unique < 200:

        agg_df = train[col].value_counts()

        plt.figure(figsize=(12,6))

        sns.barplot(agg_df.index, np.log1p(agg_df.values), alpha=0.8, color=color[0])

        plt.xlabel(col, fontsize=12)

        plt.ylabel('Log(Number of customers)', fontsize=12)

        plt.xticks(rotation='vertical')

        plt.show()

    print()

    

       