import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import gc

import psutil
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
houses = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

houses.head()
houses_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

houses_test.head()
houses.shape
houses_test.shape
#houses.info()
houses.dtypes.value_counts()
houses['SalePrice'].describe()
plt.figure(figsize =(10,8))

#histogram

sns.distplot(houses['SalePrice'])
print("Skewness: %f" % houses['SalePrice'].skew())

print("Kurtosis: %f" % houses['SalePrice'].kurt())
plt.scatter(x = houses['GrLivArea'], y = houses['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)
#Deleting outliers

train = houses.drop(houses[(houses['GrLivArea']>4000) & (houses['SalePrice']<300000)].index)

plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)
def disregardCol(df):

    l=[]

    x = df.isna().mean()

    for k,v in x.items():

        if(v>0.3):

            l.append(k)

    return [l,len(l)]

print(disregardCol(train)[0],disregardCol(train)[1],end='\n')
# Dropping Columns having more than 30% missing values of total values

train=train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)

# train.columns

train.shape
#OverallQual is a categorical column

train.OverallQual.unique()
train.OverallQual.value_counts().plot.bar()
train.OverallCond.value_counts().plot.bar()
train.OverallQual.value_counts()
plt.figure(figsize =(10,8))

sns.boxplot(x="OverallCond",y="SalePrice", data=train)
plt.figure(figsize =(10,8))

sns.boxplot(x='OverallQual', y="SalePrice", data=train)
plt.figure(figsize =(15,10))

plt.xticks(rotation=90)

sns.boxplot(x="Neighborhood", y="SalePrice", data=train)
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], height = 2.5)

plt.show()
train.columns
train['YearBuilt'].unique()
plt.figure(figsize =(10,8))

#plt.xticks(rotation=45)

sns.lineplot(y='SalePrice', x='YearBuilt', data=houses)
plt.figure(figsize =(10,8))

sns.boxplot(x='HouseStyle', y='SalePrice', data=train)
sns.lineplot(x='YearRemodAdd', y='SalePrice', data=train)
# lets drop Id because its of no use to us

train.drop("Id",1,inplace = True)
# Let's display the variables with more than 0 null values

null_cols = []

for col in train.columns:

    if train[col].isnull().sum() > 0 :

        print("Column",col, "has", train[col].isnull().sum(),"null values")    

        null_cols.append(col)
# lets visualize the null vaues

plt.figure(figsize=(12,10))

sns.barplot(x=train[null_cols].isnull().sum().index, y=train[null_cols].isnull().sum().values)

plt.xticks(rotation=45)

plt.show()
# lets check if these null values actually have any relation with the target variable



train_eda = train.copy()



for col in null_cols:

    train_eda[col] = np.where(train_eda[col].isnull(), 1, 0)  



# lets see if these null values have to do anything with the sales price

plt.figure(figsize = (16,48))

for idx,col in enumerate(null_cols):

    plt.subplot(10,2,idx+1)

    sns.barplot(x = train_eda.groupby(col)["SalePrice"].median(),y =train_eda["SalePrice"])

plt.show()
# making list of date variables

yr_vars = []

for col in train.columns:

    if "Yr" in col or "Year" in col:

        yr_vars.append(col)



yr_vars = set(yr_vars)

yr_vars
#Let's check relation of these fields with the target variable



plt.figure(figsize = (15,12))

for idx,col in enumerate(yr_vars):

    plt.subplot(2,2,idx+1)

    plt.plot(train.groupby(col)["SalePrice"].median())

    plt.xlabel(col)

    plt.ylabel("SalePrice")
#Make a note of the trend of sale price with the field "YrSold", it shows a decreasing trend which seems unreal in real state scenario, price is expected to increase as the time passes by, but here it shows opposite. Does it look right? can we do anything about it? Yes, We can Surely do!! Let's create "Age" variables out of these "Year" variables



#Let's check variations or different values present in the columns, we will start by seperating two seperate lists, one for categorical variabels and another one for numeric variables



# lets create seperate lists of categorical and numeric columns

cat_vars = []

num_vars = []

for col in train.columns.drop("SalePrice"):

    if train[col].dtypes == 'O':

        cat_vars.append(col)

    else:

        num_vars.append(col)



#lets check the lists created.

print("List of Numeric Columns:",num_vars)

print("\n")

print("List of Categorical Columns:",cat_vars)
# Let's further seperate the numeric features into continous and discrete numeric features

num_cont = []

num_disc = []

for col in num_vars:

    if train[col].nunique() > 25: # if variable has more than 25 different values, we consider it as continous variable

        num_cont.append(col)

    else:

        num_disc.append(col)
# lets check for the variance in the different continous numeric columns present in the dataset

plt.figure(figsize = (16,48))

plt.xticks(rotation=45)

for idx,col in enumerate(num_cont):

    col_bins = col+"_bins"

    train_eda[col_bins] = pd.cut(train_eda[col], 4, duplicates = 'drop') # creating bins

    plt.subplot(9,2,idx+1)

    sns.countplot(train_eda[col_bins])
# lets check for the variance in the different continous numeric columns present in the dataset

plt.figure(figsize = (16,48))

for idx,col in enumerate(num_disc):

    plt.subplot(9,2,idx+1)

    sns.countplot(train_eda[col])
# lets check for the variance in the categorical columns present in the dataset

plt.figure(figsize = (15,75))

for idx,col in enumerate(cat_vars):

    plt.subplot(22,2,idx+1)

    sns.countplot(train_eda[col])