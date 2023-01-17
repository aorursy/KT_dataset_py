# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#pre processing tools

from sklearn.impute import SimpleImputer

#from impyute.imputation.cs import fast_knn

from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.feature_selection import RFE

from tpot import TPOTRegressor

from sklearn.metrics import explained_variance_score, mean_squared_error



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#train_df.head()
#import pandas_profiling

#train_df = pd.DataFrame(train_df)

#train_df.profile_report(style={'full_width':True})
#train_df.describe()

#Removing those features which have most Nan values

train_df2 = train_df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], 1)

train_df2 = train_df2.drop(['GrLivArea', '1stFlrSF', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFinSF1', 'Id', 'LotArea'], 1)
train_df2.info()
enc = LabelEncoder()

for i in train_df2.columns:

    try:

        if np.dtype(train_df2[str(i)]) == 'object':

            train_df2[str(i)] = enc.fit_transform(train_df2[str(i)])

    except:

        pass

train_df2.info()
ndf = train_df2
ndf['BsmtCond'] = ndf['BsmtCond'].fillna('TA')

ndf['BsmtExposure'] = ndf['BsmtExposure'].fillna('No')

ndf['BsmtFinType1'] = ndf['BsmtFinType1'].fillna('Unf')

ndf['BsmtFinType2'] = ndf['BsmtFinType2'].fillna('Unf')

ndf['BsmtQual'] = ndf['BsmtQual'].fillna('TA')

ndf['GarageCond'] = ndf['GarageCond'].fillna('TA')

ndf['GarageFinish'] = ndf['GarageFinish'].fillna('Unf')

ndf['GarageQual'] = ndf['GarageQual'].fillna('TA')

ndf['GarageType'] = ndf['GarageType'].fillna('Attchd')

ndf['MasVnrType'] = ndf['MasVnrType'].fillna('None')

ndf['Electrical'] = ndf['MasVnrType'].fillna('SBrkr')

ndf.info()
enc = LabelEncoder()

for i in ndf.columns:

    try:

        if np.dtype(ndf[str(i)]) == 'object':

            ndf[str(i)] = enc.fit_transform(ndf[str(i)])

    except:

        pass

ndf.info()
imp = SimpleImputer(strategy='most_frequent')

ndf_imp = imp.fit_transform(ndf)

ndf_imp = pd.DataFrame(ndf_imp)

ndf_imp.columns = ndf.columns

ndf_imp.info()
corrmat = train_df.corr()

top_corr_features = corrmat.index()

plt.figure(figsize=(10,10))

g=sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap="plasma")
X = ndf_imp.drop('SalePrice', 1)

y = ndf_imp[['SalePrice']]
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.20, shuffle=False)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)

tpot.fit(train_x, train_y)

print("Accuracy of Regressor        {}".format(tpot.score(train_x, train_y)))




