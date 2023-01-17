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
test_data_file = '/kaggle/input/home-data-for-ml-course/test.csv'

train_data_file = '/kaggle/input/home-data-for-ml-course/train.csv'

test_data = pd.read_csv(test_data_file)

train_data = pd.read_csv(train_data_file)
from sklearn.datasets import load_boston

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
train_data.describe()

#1460 rows
#checking all the columns with null values existing

columns_na = train_data.columns[train_data.isnull().any()]
train_data[columns_na].tail()

train_data[columns_na].select_dtypes(include=['object']).describe()

 
train_data['MiscFeature'].unique()
#Using Pearson Correlation

plt.figure(figsize=(24,20))

cor = train_data.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#Correlation with output variable

cor_target = abs(cor["SalePrice"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.3]

#print(relevant_features)

relevant_features_list = relevant_features.keys()

relevant_features_list
train_data_relevant_feature = train_data[relevant_features_list]
train_data_relevant_feature_features_na = train_data_relevant_feature.columns[train_data_relevant_feature.isnull().any()]



train_data_relevant_feature_na = train_data_relevant_feature[train_data_relevant_feature_features_na]



train_data_relevant_feature_na.describe(include='all')
from sklearn.model_selection import train_test_split

X = train_data_relevant_feature.iloc[:,0:-1]

y = train_data_relevant_feature["SalePrice"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#X_test.describe()
def fillNans(df):

    df["LotFrontage"]= df.LotFrontage.fillna(df.LotFrontage.mean())

    df["MasVnrArea"]=df.MasVnrArea.fillna(df.MasVnrArea.mean())

    df["GarageYrBlt"] = df.GarageYrBlt.fillna(df.GarageYrBlt.quantile(0.5))

    df = df.fillna(df.mean())

    return df
X = fillNans(X)

X_train = fillNans(X_train)

X_test = fillNans(X_test)
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV



tree_clf = DecisionTreeRegressor()

param_grid= {'criterion':['mae'],'max_depth':[10,15],'min_samples_split':[30, 35], 'max_leaf_nodes': [50,100,150],'min_samples_leaf': [20,30,40]}

grid_search = GridSearchCV(tree_clf, param_grid, cv=5, verbose=3)

grid_search.fit(X_train, y_train)





#tree_clf.fit(X_train,y_train)

#tree_clf.fit(X,y)
grid_search.best_params_
grid_search.best_score_
#y_pred = tree_clf.predict(X_test)

y_pred = grid_search.predict(X_test)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error

#cross_val_score(tree_clf, X, y, cv=10)



MAE = mean_absolute_error(y_pred, y_test)

print(MAE)

#score = tree_clf.score(X_test, y_test)

#print(score)
tree_clf_full_data= DecisionTreeRegressor(criterion='mae',max_depth= 10,max_leaf_nodes= 150,min_samples_leaf= 20,min_samples_split= 30)

tree_clf_full_data.fit(X,y)
test_data_na = fillNans(test_data[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',

       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',

       'GarageArea', 'WoodDeckSF', 'OpenPorchSF']])



test_preds = tree_clf_full_data.predict(test_data_na)
output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)