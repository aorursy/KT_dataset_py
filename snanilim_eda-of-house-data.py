import numpy as np

import pandas as pd

import scipy.stats as st

pd.set_option('display.max_columns', None)



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set_style('whitegrid')



import missingno as msno



from sklearn.preprocessing import StandardScaler

from scipy import stats







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train_data.head()
train_data.shape
train_data = train_data.drop(columns=['Id'])
total_rows = train_data.shape[0]

total_columns = train_data.shape[1]

print('Total rows', total_rows)

print('Total columns', total_columns)
all_columns = train_data.columns

all_columns
train_data['SalePrice'].sort_values(ascending=False).head()
train_data['SalePrice'].sort_values(ascending=True).head()
len(train_data['SalePrice'].unique())
train_data['SalePrice'].value_counts().head(20)
train_data.info()
train_data.describe()
train_data.describe(include=['object', 'bool'])
print(train_data['SalePrice'].mean())

print('-' * 20)

print(train_data['SalePrice'].median())

print('-' * 20)

print('Difference between mean and median', train_data['SalePrice'].mean() - train_data['SalePrice'].median())
train_data.skew()
train_data.kurt()
plt.figure(figsize=(12, 7))

sns.distplot(train_data.skew(),color='green',axlabel ='Skewness')

plt.show()
plt.figure(figsize=(12, 7))

sns.distplot(train_data.kurt(),color='orange',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)

plt.show()
total = train_data.isnull().sum().sort_values(ascending=False)

percent = ((train_data.isnull().sum() / total_rows) * 100).sort_values(ascending=False)

null_data = pd.concat([total, percent], axis=1,join='outer', keys=['Null count', 'Percentage %'])

null_data.index.name ='Columns'

null_data = null_data[null_data['Null count'] > 0].reset_index()

null_data
null_columns = null_data['Columns']

null_columns
null_rows = train_data.isnull().sum(axis=1).sort_values(ascending=False).head(20)

# null_rows = null_rows.head(20)

null_rows
null_rows.index
((null_rows *100) / 80)
train_data.loc[null_rows.index].head()
# msno.bar.__code__.co_varnames

msno.bar(train_data.sample(1460), labels=True, fontsize=12)
msno.heatmap(train_data)
msno.dendrogram(train_data)
copy_data = train_data.copy()
for column in null_columns:

    copy_data[column] = np.where(train_data[column].isnull(), 1, 0)

    

    plot_data = copy_data.groupby(by=[column])['SalePrice'].median()

    plot_data = pd.DataFrame(plot_data)

    plot_data = plot_data.reset_index()

    sns.barplot(x=plot_data[column], y=plot_data['SalePrice'], data=plot_data, palette="Blues_d")

    

    plt.xticks(plot_data[column], ('value(0)', 'Null(1)'))

    plt.xlabel(column)

    plt.ylabel('SalePrice')

    plt.show()
numerical_data = train_data.select_dtypes(include=[np.number])

numerical_columns = numerical_data.columns

numerical_columns
discrete_column = []

continious_column = []

year_column = []
for column in numerical_columns:

    if 'Year' in column or 'Yr' in column:

#         print(column)

        year_column.append(column)
for column in numerical_columns:

    if column != 'SalePrice' and column not in year_column:

        if len(train_data[column].unique()) < 25:

            discrete_column.append(column)

        else:

            continious_column.append(column)
discrete_column
continious_column
categorical_data = train_data.select_dtypes(include=[np.object])

categorical_columns = categorical_data.columns

categorical_columns
year_column = []
for column in numerical_columns:

    if 'Year' in column or 'Yr' in column:

#         print(column)

        year_column.append(column)
train_data[year_column]
for column in year_column:

    plt.figure(figsize=(10, 7))

    train_data.groupby(by=[column])['SalePrice'].median().plot(color = ['c', 'y'])

    plt.show()
year_data = train_data.copy()


for column in year_column:

    if column != 'YrSold':

        plt.figure(figsize=(10, 7))

        year_data[column] = year_data['YrSold']-year_data[column]

        

        sns.scatterplot(x=year_data[column], y=year_data['SalePrice'], data=year_data)

        plt.xlabel(column)

        plt.ylabel('SalePrice')

        plt.show()
for column in continious_column:

    plt.figure(figsize=(10, 7))

    train_data[column].plot.hist(color = "skyblue")

    plt.xlabel(column)

    plt.show()
copy_data = train_data.copy()
for column in continious_column:

    if 0 in copy_data[column].unique():

        pass

    else:

#         print(column)

        plt.figure(figsize=(10, 7))

        con_data = np.log(copy_data[column])

        con_data.plot.hist(color = "skyblue")

        plt.xlabel(column)

        plt.show()
for column in discrete_column:

    plt.figure(figsize=(10, 7))

    train_data.groupby(by=[column])['SalePrice'].median().plot.bar(color = "skyblue")

    

    plt.show()
for column in continious_column:

    if 0 in train_data[column].unique():

        pass

    else:

        plt.figure(figsize=(10, 7))

        sns.scatterplot(x=train_data[column], y=train_data['SalePrice'], data=train_data)

        

        plt.xlabel(column)

        plt.ylabel('SalePrice')

        plt.show()
copy_data = train_data.copy()
copy_data['SalePrice'] = np.log(copy_data['SalePrice'])

for column in continious_column:

    if 0 in copy_data[column].unique():

        pass

    else:

        plt.figure(figsize=(10, 7))

        copy_data[column] = np.log(copy_data[column])

        sns.scatterplot(x=copy_data[column], y=copy_data['SalePrice'], data=copy_data)

        

        plt.xlabel(column)

        plt.ylabel('SalePrice')

        plt.show()
for column in categorical_columns:

    plt.figure(figsize=(10, 7))

    copy_data.groupby(by=column)['SalePrice'].median().plot.bar(color = "skyblue")

    

    plt.xlabel(column)

    plt.ylabel('SalePrice')

    plt.show()
var = 'OverallQual'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
feature_corr = numerical_data.corr()

feature_corr['SalePrice'].sort_values(ascending=False)
plt.figure(figsize=(15, 12))

sns.heatmap(feature_corr,square = True,  vmax=0.8)
k = 11

cols = feature_corr.nlargest(k, 'SalePrice')['SalePrice'].index

cols

cm = np.corrcoef(train_data[cols].values.T) # transformed data

cm

f , ax = plt.subplots(figsize = (14,12))

sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True, linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)

columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']

sns.pairplot(train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')

plt.show()
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:,np.newaxis]);

# saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10] # specifies that you want to slice out a 1D vector of length 97 from a 2D array.

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
# copy_data[column] = np.log(copy_data[column])

plt.figure(figsize=(10, 7))

train_data.boxplot(column='SalePrice', notch=True, vert=False)



plt.show()
copy_data = train_data.copy()
copy_data['SalePrice'] = np.log(copy_data['SalePrice'])

plt.figure(figsize=(10, 7))

copy_data.boxplot(column='SalePrice', notch=True, vert=False)



plt.show()
for column in continious_column:

    if 0 in copy_data[column].unique():

        pass

    else:

        plt.figure(figsize=(10, 7))

        copy_data[column] = np.log(copy_data[column])

        copy_data.boxplot(column=column , notch=True, vert=False)

        

        plt.show()
y = train_data['SalePrice']



plt.figure(figsize=(10, 7))

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=True, fit=st.johnsonsu, color='#636efa')



plt.figure(figsize=(10, 7))

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=True, fit=st.norm, color='#636efa')



plt.figure(figsize=(10, 7))

plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=True, fit=st.lognorm, color='#636efa')
plt.figure(figsize=(10, 7))

res = stats.probplot(train_data['SalePrice'], plot=plt)
copy_data = train_data.copy()
#applying log transformation

plt.figure(figsize=(10, 7))

copy_data['SalePrice'] = np.log(copy_data['SalePrice'])

res = stats.probplot(copy_data['SalePrice'], plot=plt)
print("Skewness:", train_data['SalePrice'].skew())

print('-' * 30)

print("Kurtosis:", train_data['SalePrice'].kurt())
# plt.figure(figsize=(5, 3))

plt.figure(figsize=(10, 7))

plt.hist(train_data['SalePrice'],orientation = 'vertical',histtype = 'bar', color ='#21bf73')

plt.show()
# plt.figure(figsize=(5, 3))

plt.figure(figsize=(10, 7))

target = np.log(train_data['SalePrice'])

target.skew()

plt.hist(target,color='#21bf73')