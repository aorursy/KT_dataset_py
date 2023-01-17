# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
item_sales = pd.read_csv('../input/Train_UWu5bXk.csv')

item_sales
item_sales.isna().sum()/item_sales.count()*100
item_sales = item_sales.drop('Outlet_Size', axis=1)

item_sales
item_sales['Item_Weight'] = item_sales['Item_Weight'].fillna(item_sales['Item_Weight'].mean())
item_sales['Item_Weight'].isna().sum()
item_sales.var()
item_sales.corr()
item_sales_reduced = item_sales[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year', 'Item_Outlet_Sales']]

item_sales_reduced
from sklearn.preprocessing import StandardScaler

wieght_transformer = StandardScaler()

weights = wieght_transformer.fit_transform(item_sales_reduced.values)

weights
Y = weights[:, -1]

X = weights[:, :-1]
Y.shape
X.shape
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

lr_model.fit(X, Y)
lr_model.coef_
# BFE will be HW
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

dataset = load_iris()

X_iris = dataset['data']

Y_iris = dataset['target']
dataset.keys()
pca_model = PCA()

pca_model.fit(X)
pca_model.components_
pca_model.explained_variance_
import seaborn as sns

import matplotlib.pyplot as plt
sns.pairplot(item_sales_reduced)
pca_model_iris = PCA(n_components=2)

pca_model_iris.fit(X_iris)
pca_model_iris.explained_variance_
pca_model_iris.components_
sns.pairplot(pd.DataFrame(X_iris, columns=dataset['feature_names']))