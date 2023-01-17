import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score,KFold

from sklearn.metrics import mean_squared_error

from sklearn.tree import _tree

from sklearn import linear_model

from sklearn.preprocessing import LabelEncoder, MinMaxScaler



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print("Train Shape: {}".format(train.shape))

print("Test Shape: {}".format(test.shape))
test = pd.read_csv('../input/test.csv')

print("Test Shape: {}".format(test.shape))
train["LogSalePrice"] = np.log1p(train['SalePrice'])



print("Mean(std) of Sale Price: {0:.0f}({1:.0f})".format(train["SalePrice"].mean(), train["SalePrice"].std()))

print("Mean(std) of Log Sale Price: {:.2f}({:.2f})".format(train["LogSalePrice"].mean(), train["LogSalePrice"].std()))



f, axes = plt.subplots(1, 2, figsize=(12,4))

ax1 = sns.distplot(train.SalePrice, ax=axes[0])

ax2 = sns.distplot(train.LogSalePrice, ax=axes[1])

ax1.set(xlabel='Sale Price', ylabel='Proportion', title='Sale Price')

ax2.set(xlabel='Log Sale Price', ylabel='Proportion', title='Log Sale Price')

plt.show()
print("Numerical Variables:")

train.select_dtypes(exclude=['object']).columns.values
area_vars = ["LogSalePrice","LotArea","TotalBsmtSF",'1stFlrSF','2ndFlrSF','GrLivArea','GarageArea']

area_train = train[area_vars]

area_train.dropna()

sns.pairplot(area_train)
f, axes = plt.subplots(1, 2, figsize=(12,4))

ax1 = sns.violinplot(x="OverallQual", y="LogSalePrice", data=train, ax=axes[0])

ax2 = sns.violinplot(x="OverallCond", y="LogSalePrice", data=train, ax=axes[1])

ax1.set(xlabel='Quality', ylabel='Log Sale Price', title='Overall Quality')

ax2.set(xlabel='Condition', ylabel='Log Sale Price', title='Overall Condition')

plt.show()
plt.figure(figsize=(20,8))

ax = sns.boxplot(x="YearBuilt", y="LogSalePrice", data=train)

ax.set_xlabel(xlabel='Year Sold')

ax.set_ylabel(ylabel='Log Sale Price')

ax.set_title(label='House Price by Year Sold')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,8))

ax = sns.boxplot(x="YearRemodAdd", y="LogSalePrice", data=train)

ax.set_xlabel(xlabel='Year Renovated')

ax.set_ylabel(ylabel='Log Sale Price')

ax.set_title(label='House Price by Year Renovated')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(6,4))

ax = sns.boxplot(x="MoSold", y="LogSalePrice", data=train)

ax.set_xlabel(xlabel='Sale Month')

ax.set_ylabel(ylabel='Log Sale Price')

ax.set_title(label='Month of Sale')

plt.show()
print("Categorical Variables:")

train.select_dtypes(include=['object']).columns.values
# Function to plot categical data against the target variable "SalePrice"

def category_boxplot(table, var):

    grouped = table.groupby(var)['SalePrice'].mean().sort_values(ascending=False)

    sns.boxplot(x=var, y='LogSalePrice', data=table, order=grouped.index)

    

category_boxplot(train, "SaleCondition")
train['MSSubClass'] = train['MSSubClass'].apply(str)

train['OverallCond'] = train['OverallCond'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)
# Function to plot missing data percentage

def missing_plot(table):

    f, ax = plt.subplots()

    plt.xticks(rotation='90')

    sns.barplot(x=table.index, y=table)

    plt.xlabel('Features')

    plt.ylabel('Percentage of Missing Values')

    plt.title('Percentage of Missing Data by Features')



train_miss = (train.isnull().sum() / len(train)) * 100

train_miss = train_miss.drop(train_miss[train_miss == 0].index).sort_values(ascending=False)

missing_plot(train_miss)
train_drop = train_miss.drop(train_miss[train_miss < 50].index).index.values

train_drop
print("Pre Drop Shape: {}".format(train.shape))



for i, col in enumerate(train_drop):

    if i==1:

        train2 = train.drop(col, axis=1)

    elif i>1:

        train2 = train2.drop(col, axis=1)

    

print("Post Drop Shape: {}".format(train2.shape))
cats = train2.select_dtypes(include=['object'])

cols = cats.columns.values

df2=pd.DataFrame([0], columns=['count'], index=['Test'])



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(cats[c].values))

    cats[c] = lbl.transform(list(cats[c].values))

    df = pd.DataFrame([len(cats[c].unique())], columns=['count'], index=[c])

    df2 = df2.append(df)

df2 = df2.drop(df2[df2['count'] == 0].index)

df2 = df2.sort_values(by=['count'], ascending=False)
