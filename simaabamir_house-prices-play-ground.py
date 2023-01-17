# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

%matplotlib inline



import warnings

warnings.filterwarnings("ignore",category=FutureWarning)

warnings.filterwarnings("ignore", category=DeprecationWarning)



#3ways to access system files

from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))

print(check_output(["ls", "/kaggle/input/house-prices-advanced-regression-techniques/"]).decode("utf8"))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# setting the number of cross validations used in the Model part 

nr_cv = 5



# switch for using log values for SalePrice and features     

use_logvals = 1    

# target used for correlation 

target = 'SalePrice_Log'

    

# only columns with correlation above this threshold value  

# are used for the ML Regressors in Part 3

min_val_corr = 0.4    

    

# switch for dropping columns that are similar to others already used and show a high correlation to these     

drop_similar = 1
def get_best_score(grid):

    """Function to return best score

    args : grid

    output : best_score"""

    

    

    best_score = np.sqrt(-grid.best_score_)

    print(best_score)    

    print(grid.best_params_)

    print(grid.best_estimator_)

    

    return best_score
def print_cols_large_corr(df,nr_c,targ):

    """

    Function to print columns with larger correlations

    args:

        df = dataframe

        nr_c = num of columns

        targ = target column

    """

    corr=df.corr()

    corr_abs=corr.abs()

    print(corr_abs.nlargest(nr_c, targ)[targ])
def plot_corr_matrix(df, nr_c, targ) :

    """

    Function to plot correlation matrix between variables and target

    args:

        df = dataframe

        nr_c = num of columns

        targ = target column

    """

    

    corr = df.corr()

    corr_abs = corr.abs()

    cols = corr_abs.nlargest(nr_c, targ)[targ].index

    cm = np.corrcoef(df[cols].values.T)



    plt.figure(figsize=(nr_c/1.5, nr_c/1.5))

    sns.set(font_scale=1.25)

    sns.heatmap(cm, linewidths=1.5, annot=True, square=True, 

                fmt='.2f', annot_kws={'size': 10}, 

                yticklabels=cols.values, xticklabels=cols.values

               )

    plt.show()
df_train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(df_train.shape)

df_train.info()
df_train.head()
df_test.head()
df_train.describe()
df_test.describe()
print(df_test.shape)

df_test.info()
print(df_train.shape)

print(df_test.shape)
sns.distplot(df_train['SalePrice']);

#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
# going from non-normal to logarithmic distribution

df_train['SalePrice_Log']=np.log(df_train['SalePrice'])

sns.distplot(df_train['SalePrice_Log']);

#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice_Log'].skew())

print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())

# dropping old SalePrice column

df_train.drop('SalePrice',axis=1,inplace=True)
numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index

print("Number of Numerical features: ", len(numerical_feats))



categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index

print("Number of Categorical features: ", len(categorical_feats))
print('Numerical columns in train set :')

print(df_train[numerical_feats].columns)

print("*"*100)

print('Categorical columns in train set: ')

print(df_train[categorical_feats].columns)
df_train[numerical_feats].head()
df_train[categorical_feats].head()
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# unique values in Pool

df_train.PoolQC.unique()
# Columns which have NaN values present in them

nan_cols = [i for i in df_train.columns if df_train[i].isnull().any()]

print(len(nan_cols))

nan_cols
# columns where NaN values have meaning e.g. no pool etc.

cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',

               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',

               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',

               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',

               'MSZoning', 'Utilities']

len(cols_fillna)
# replace 'NaN' with 'None' in these columns

for col in cols_fillna:

    df_train[col].fillna('None',inplace=True)

    df_test[col].fillna('None',inplace=True)
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(5)
# df_train['LotFrontage'].head()

cols=['LotFrontage','GarageYrBlt','MasVnrArea','SalePrice_Log','ExterCond']

for col in cols:

    print(df_train[i].dtype)
# fillna with mean for the remaining columns: LotFrontage, GarageYrBlt, MasVnrArea

df_train.fillna(df_train.mean(), inplace=True)

df_test.fillna(df_test.mean(), inplace=True)
total=df_train.isnull().sum().sort_values(ascending=False)

percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])

missing_data.head()





# total = df_train.isnull().sum().sort_values(ascending=False)

# percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# missing_data.head(5)
#Now, we should have 0 Nan columns

# Columns which have NaN values present in them

nan_cols = [i for i in df_train.columns if df_train[i].isnull().any()]

print(len(nan_cols))

nan_cols
df_train.isnull().sum().sum()
df_test.isnull().sum().sum()
numerical_feats
categorical_feats
len(numerical_feats)+len(categorical_feats)
for col in numerical_feats:

    print('{:15}'.format(col), 

          'Skewness: {:05.2f}'.format(df_train[col].skew()) , 

          '   ' ,

          'Kurtosis: {:06.2f}'.format(df_train[col].kurt())  

         )
# lets check skewness and kurtosis for GrLivArea

sns.distplot(df_train['GrLivArea']);

#skewness and kurtosis

print("Skewness: %f" % df_train['GrLivArea'].skew())

print("Kurtosis: %f" % df_train['GrLivArea'].kurt())
sns.distplot(df_train['LotArea']);

#skewness and kurtosis

print("Skewness: %f" % df_train['LotArea'].skew())

print("Kurtosis: %f" % df_train['LotArea'].kurt())
# transforming to make closer to normal dstribution (log) for GrLivArea and LotArea

for df in [df_train, df_test]:

    df['GrLivArea_Log'] = np.log(df['GrLivArea'])

    df.drop('GrLivArea', inplace= True, axis = 1)

    df['LotArea_Log'] = np.log(df['LotArea'])

    df.drop('LotArea', inplace= True, axis = 1)

    

    

    

numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index

print(len(numerical_feats))

numerical_feats
sns.distplot(df_train['GrLivArea_Log']);

#skewness and kurtosis

print("Skewness: %f" % df_train['GrLivArea_Log'].skew())

print("Kurtosis: %f" % df_train['GrLivArea_Log'].kurt())
sns.distplot(df_train['LotArea_Log']);

#skewness and kurtosis

print("Skewness: %f" % df_train['LotArea_Log'].skew())

print("Kurtosis: %f" % df_train['LotArea_Log'].kurt())
nr_rows = 12

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



li_num_feats = list(numerical_feats)

li_not_plot = ['Id', 'SalePrice', 'SalePrice_Log']

li_plot_num_feats = [c for c in list(numerical_feats) if c not in li_not_plot]





for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_plot_num_feats):

            sns.regplot(df_train[li_plot_num_feats[i]], df_train[target], ax = axs[r][c])

            stp = stats.pearsonr(df_train[li_plot_num_feats[i]], df_train[target])

            #axs[r][c].text(0.4,0.9,"title",fontsize=7)

            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])

            axs[r][c].set_title(str_title,fontsize=11)

            

plt.tight_layout()    

plt.show()   
sns.distplot(df_train['OverallQual'])
df_train = df_train.drop(

    df_train[(df_train['OverallQual']==10) & (df_train['SalePrice_Log']<12.3)].index)
df_train = df_train.drop(

    df_train[(df_train['GrLivArea_Log']>8.3) & (df_train['SalePrice_Log']<12.5)].index)
# columns where NaN values have meaning e.g. no pool etc.

cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',

               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',

               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',

               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',

               'MSZoning', 'Utilities']
df_train.info()
df_train.isna()
numerical_columns_train
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_test= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_train= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df_test.head()
df=df_train

df.head()
df.shape
df.info()
# Analyse data

# clean data

# take significant columns

# make base model

# apply feature engineering

# try to beat model , use many model

# predict