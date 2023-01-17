## Importing the Libraries

import os

import pandas as pd

pd.set_option('max_columns',105)

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats as ss

from IPython.core.display import HTML

import warnings

warnings.filterwarnings("ignore")

sns.set()
## Loading the Dataset

train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
## Printing the shape of the dataset

print("Shape of the Train dataset is:",train_data.shape)

print("*"*50)

print("Shape of the Test dataset is:",test_data.shape)
train_data.head()
## Summary of numeric values in the train dataset

train_data.describe()
## Printing the datatype of train and test set

print("Training data:",train_data.info())

print("*"*50)

print("Test data:",test_data.info())
## Distribution of Traget varible i.e. "SalePrice"

sns.distplot(train_data['SalePrice'])



## Skewness and Kurtosis

print("Skewness %f" % train_data['SalePrice'].skew())

print("Kurtosis %f" % train_data['SalePrice'].kurt())
## Fitting log curve to check the distribution with Log Transfornation

train_data['SalePrice_Log'] = np.log(train_data['SalePrice'])

## Log Transformed Curve

sns.distplot(train_data['SalePrice_Log'])



## Skewness and Kurtosis of the data

print("Skewness",train_data['SalePrice_Log'].skew())

print("kurtosis",train_data['SalePrice_Log'].kurt())
## Numerical and Ctegorical columns

numerical_features = train_data.dtypes[train_data.dtypes != "object"].index

print("Total Numerical Features in the dataset:",len(numerical_features))



categorical_features = train_data.dtypes[train_data.dtypes == "object"].index

print("Total Categorical Features in the dataset:", len(categorical_features))
print("Numerical columns:",numerical_features)

print("*"*60)

print("Categrical features:",categorical_features)
## List of Features with Missing Values

total = train_data.isnull().sum().sort_values(ascending = False)

percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ["Total","Percent"])

missing_data.head(20)
## Filling the Missing Values where  NAN has meaning ex. No pool 

## This information can be found from the Data Discription

fill_na_cols = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',

               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',

               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',

               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',

               'MSZoning', 'Utilities']



## Replace 'NaN' with 'None' in these columns

for cols in fill_na_cols:

    train_data[cols].fillna('None',inplace = True)

    test_data[cols].fillna('None',inplace = True)
## Checking the percentage of the missing values after this transformation

total = train_data.isnull().sum().sort_values(ascending = False)

percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total,percent],axis = 1, keys = ["Total","Percent"])

missing_data.head(10)
## Filling the remainig numerical columns with the mean value

train_data.fillna(train_data.mean(), inplace = True)

test_data.fillna(test_data.mean(),inplace = True)
## Checking the missing data status

print("Missing Status of the Training data:",train_data.isnull().sum().sum())

print("Missing Status of the Test data:",test_data.isnull().sum().sum())
## Checking the Distribution of the columns in the Train dataset

for col in numerical_features:

    print('{:15}'.format(col),

          'Skweness:{:05.2f}'.format(train_data[col].skew()),

          '   ',

          'Kurtosis: {:06.2f}'.format(train_data[col].kurt())

         )
train_data.head()
test_data.columns
sns.distplot(train_data['LotArea'])

print("Skewness: %f" %train_data['LotArea'].skew())

print("Kurtosis: %f" %train_data['LotArea'].kurt())
sns.distplot(train_data['GrLivArea'])

print("Skewness of the data:",train_data['GrLivArea'].skew())

print("Kurtosis of the data:",train_data['GrLivArea'].kurt())
for data in [train_data, test_data]:

    data['GrLivArea_log'] = np.log(data['GrLivArea'])

    data.drop('GrLivArea', inplace= True, axis = 1)

    data['LotArea_log'] = np.log(data['LotArea'])

    data.drop('LotArea', inplace= True, axis = 1)

    

    

    

numerical_feats = train_data.dtypes[train_data.dtypes != "object"].index

   
sns.distplot(train_data['LotArea_log'])

print("Skewness: %f" %train_data['LotArea_log'].skew())

print("Kurtosis: %f" %train_data['LotArea_log'].kurt())
sns.distplot(train_data['GrLivArea_log'])

print("Skewness of the data:",train_data['GrLivArea_log'].skew())

print("Kurtosis of the data:",train_data['GrLivArea_log'].kurt())
target = 'SalePrice_Log'
## Separating Numerical and Categorical features after "Log" Transformation

numerical_features = train_data.dtypes[train_data.dtypes != "object"].index

categorical_features = train_data.dtypes[train_data.dtypes == "object"].index
n_rows = 12

n_cols = 3



fig,axs = plt.subplots(n_rows, n_cols, figsize = (n_cols*3.5,n_rows*3))



num_features_list = list(numerical_features)

not_to_plot_list = ['Id','SalePrice','SalePrice_Log']

plot_numerical_features_list = [c for c in list(numerical_features) if c not in not_to_plot_list]



for r in range(0, n_rows):

    for c in range(0, n_cols):

        i = r*n_cols + c

        if i < len(plot_numerical_features_list):

            sns.regplot(train_data[plot_numerical_features_list[i]], train_data[target], ax = axs[r][c])

            stp = ss.pearsonr(train_data[plot_numerical_features_list[i]], train_data[target])

            str_title = "r = " + "{0:.2f}".format(stp[0]) + " " "p = " + "{0:.2f}".format(stp[1])

            axs[r][c].set_title(str_title, fontsize = 11)



plt.tight_layout()

plt.show()
for cat in list(categorical_features):

    print(train_data[cat].value_counts())

    print('#'*50)
cat_features_list = list(categorical_features)

n_rows = 15

n_cols = 3



fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * 4, n_rows * 3))



for r in range(0, n_rows):

    for c in range(0, n_cols):

        i = r * n_cols + c

        if i < len(cat_features_list):

            sns.boxplot(x = cat_features_list[i], y = target, data = train_data, ax = axs[r][c])



plt.tight_layout()

plt.show()
#correlation matrix

corrmat = train_data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);