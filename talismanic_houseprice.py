# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Loading the data



train_df_orig = pd.read_csv("../input/train.csv")

test_df_orig = pd.read_csv("../input/test.csv")





# train_df_dummies = pd.get_dummies(data = train_df_orig, columns =['MSZoning','Alley','LotShape', 'LandContour', 'Utilities', 'LandSlope', 'Neighborhood', 'Condition1','Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd','MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'])

# train_df_dummies = train_df_dummies.drop(["Street","LotConfig"], axis = 1)

# train_df_dummies.head()



# test_df_dummies = pd.get_dummies(data = test_df_orig, columns =['MSZoning','Alley','LotShape', 'LandContour', 'Utilities', 'LandSlope', 'Neighborhood', 'Condition1','Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd','MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'])

# test_df_dummies = test_df_dummies.drop(["Street","LotConfig"], axis = 1)



#sns.distplot(train_df_orig.SalePrice.dropna())





#train_df_orig.head()
#Heatmap to identify features

fig = plt.subplots(figsize=(30,20))

sns.heatmap(train_df_orig.corr(), annot = True, cmap = 'plasma')
# Identified features are 

# OverallQual, GrLivArea, GarageArea, TotalBsmtSF, FullBath, YearBuilt, YearRemodAdd, WoodDeckSF, OpenPorchSF



#Checking other categorical data

g = sns.factorplot(x='MSZoning', y ='SalePrice', data = train_df_orig, kind = 'swarm' )

g.set_xticklabels( rotation = -45)
g = sns.factorplot(x='LandContour', y ='SalePrice', data = train_df_orig, kind = 'swarm' )

g.set_xticklabels( rotation = -45)
#factorplotting the categorical values

g = sns.factorplot(x='Neighborhood', y ='SalePrice', data = train_df_orig, kind = 'swarm' )

g.set_xticklabels( rotation = -45)
X_train = train_df_orig[["OverallQual", "GrLivArea", "GarageArea", "TotalBsmtSF", "FullBath", "YearBuilt", "YearRemodAdd", "WoodDeckSF", "OpenPorchSF"]]

X_test = test_df_orig[["OverallQual", "GrLivArea", "GarageArea", "TotalBsmtSF", "FullBath", "YearBuilt", "YearRemodAdd", "WoodDeckSF", "OpenPorchSF"]]

Y_train = train_df_orig['SalePrice']



#filling missing values



from sklearn.preprocessing import Imputer

my_imputer = Imputer()

X_train = my_imputer.fit_transform(X_train)

X_test = my_imputer.fit_transform(X_test)



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
#KNN

knn = KNeighborsClassifier(n_neighbors = 2)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn