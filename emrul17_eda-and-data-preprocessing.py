import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

import os

print(os.listdir("../input"))
# Loading  all available datasets

df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')

df_train.head()
# Function to print the basic information of the data

def data_info(df):



    print('Shape of the data: ', df.shape)

    

    print('------------########################------------------')

    print('                                                     ')

    print('Information of the data:')

    print(' ', df.info())

    

    print('------------########################------------------')

    print('                                                     ')

    print('Check the duplication of the data:', df.duplicated().sum())
data_info(df_train)
# Function find out the Statistical susmmary 

def summary(df):

    print('\n Statistical Summary of Numberical data:\n', df.describe(include=np.number))

    print('------------########################------------------')

    print('\n Statistical Summary of categorical data:\n',df.describe(include='O'))

    

summary(df_train)
# A function for calculating the missing data

def missing_data(df):

    tot_missing=df.isnull().sum().sort_values(ascending=False)

    Percentage=tot_missing/len(df)*100

    missing_data=pd.DataFrame({'Missing Percentage': Percentage})

    

    return missing_data.head(20)



missing_data(df_train)
# Features with missing value

miss_col1=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageType', 'GarageYrBlt',

           'GarageFinish', 'GarageQual', 'BsmtExposure','BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 

           'MasVnrArea', 'MasVnrType']

# Imputing missing value

for col in miss_col1:

    if df_train[col].dtype=='O':

        df_train[col]=df_train[col].fillna("None")

    else:

        df_train[col]=df_train[col].fillna(0)
# Imputing missing value with neighborhood value

df_train['LotFrontage']=df_train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# Imputing missing value with mode

df_train['Electrical']=df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])
missing_data(df_train)
# Let's separate the numerical and categorical columns

numerical_col=df_train.select_dtypes(include=[np.number])

categorical_col=df_train.select_dtypes(include=[np.object])

num_var=numerical_col.columns.tolist()

cat_var=categorical_col.columns.tolist()
# Boxplot for target

plt.figure(figsize=(12,8))

sns.boxplot(df_train['SalePrice'])
# Remove outliers from target variables

df_train=df_train[df_train['SalePrice']<700000]

df_train.head()
# Distribution plot

plt.figure(figsize=(12,8))

sns.distplot(df_train['SalePrice'])
# Distribution plot

sns.distplot(df_train['SalePrice'] , fit=norm);



# Probability parameter

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
#Log tranformation of target column

df_train["SalePrice"] = np.log(df_train["SalePrice"])



#Plot the new distriution

sns.distplot(df_train['SalePrice'] , fit=norm);



# probability parameter for normal distribution

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)])

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

# Outliers Check

def outlier(df):

    stat=df.describe()

    IQR=stat['75%']-stat['25%']

    upper=stat['75%']+1.5*IQR

    lower=stat['25%']-1.5*IQR

    print('The upper and lower bounds for outliers are {} and {}'.format(upper,lower))
outlier(df_train['SalePrice'])
# Function to plot target vs categorical data

def cat_plot(df):

    for col in cat_var:

        f, ax = plt.subplots(figsize=(12, 6))

        sns.boxplot(x=col,y='SalePrice', data=df)

        plt.xlabel(col)

        plt.title('{}'.format(col))



cat_plot(df_train)
# Function to plot target vs numerical data

def num_plot(df):

    for col in num_var:

        f, ax = plt.subplots(figsize=(12, 6))

        plt.scatter(x=col,y='SalePrice', data=df)

        plt.xlabel(col)

        plt.ylabel("SalePrice")

        plt.title('{}'.format(col))
num_plot(numerical_col)
# Removing suspicious outliers

df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)

df_train=df_train.drop(df_train[(df_train['LotFrontage']>250) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)

df_train=df_train.drop(df_train[(df_train['BsmtFinSF1']>1400) & (df_train['SalePrice']<400000)].index).reset_index(drop=True)

df_train=df_train.drop(df_train[(df_train['TotalBsmtSF']>5000) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)

df_train=df_train.drop(df_train[(df_train['1stFlrSF']>4000) & (df_train['SalePrice']<300000)].index).reset_index(drop=True)
new_cat=['GrLivArea','LotFrontage','BsmtFinSF1','TotalBsmtSF','1stFlrSF']
# Plotting after removing outliers

for col in new_cat:

        f, ax = plt.subplots(figsize=(12, 6))

        plt.scatter(x=col,y='SalePrice', data=df_train)
corr= df_train.corr()

f, ax = plt.subplots(figsize=(14, 9))

sns.heatmap(corr, vmax=.8, square=True)
#saleprice correlation matrix

k = 20 #number of variables for heatmap

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

f , ax = plt.subplots(figsize = (14,12))

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True,linewidths=0.004, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

# Total area in units of square feet

df_train['TotSF']=df_train['TotalBsmtSF']+df_train['1stFlrSF']+df_train['2ndFlrSF']

df_train['TotArea']=df_train['GarageArea']+df_train['GrLivArea']
plt.scatter(x='TotArea',y='SalePrice', data=df_train)
#df_train['TotSF']=df_train[df_train['TotSF']>8500]
plt.scatter(x='TotSF',y='SalePrice', data=df_train)
cols=['MSSubClass','OverallCond','YrSold','MoSold']



for col in cols:

    df_train[col] = df_train[col].apply(str)
categorical_col=df_train.select_dtypes(include=[np.object])

new_catcol=categorical_col.columns

new_catcol
ordinal_cat=['OverallCond','KitchenQual','YrSold','MoSold','Fence','PoolQC','FireplaceQu','GarageQual', 

             'GarageCond','LotShape','LandSlope','HouseStyle','ExterQual','ExterCond','BsmtQual', 

             'BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2','HeatingQC','KitchenQual','CentralAir',

             'MSSubClass']



# label Encoding for ordinal data

from sklearn.preprocessing import LabelEncoder

label_encode=LabelEncoder()



for col in ordinal_cat:

    df_train[col]=label_encode.fit_transform(df_train[col])
df_train.select_dtypes(include=[np.object]).head()
# One hot encoding for nominal data

df_train=pd.get_dummies(df_train)
df_train=df_train.drop(columns=['Id','SalePrice'])
# Final train clean dataset

df_train.head()