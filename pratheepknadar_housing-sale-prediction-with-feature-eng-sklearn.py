# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
b = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

a = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
discrete = [var for var in a.columns if a[var].dtype!='O' and var!='SalePrice' and var!='Id' and a[var].nunique()<11]

continuous = [var for var in a.columns if a[var].dtype!='O' and var!='SalePrice'and var!='Id' and var not in discrete]



# categorical

categorical = [var for var in a.columns if a[var].dtype=='O' and var!='Id']

print(discrete)

print()

print(continuous)

print()

print(categorical)

a.head()
b.head()
pd.crosstab(b['LotFrontage'],b['Street'],normalize = True)
from sklearn.pipeline import Pipeline

!pip install feature_engine

import feature_engine.missing_data_imputers as mdi

from feature_engine import categorical_encoders as ce

pipe = Pipeline([

     ('indicator',mdi.AddMissingIndicator(

    variables=discrete)),

    ('median_imputer',mdi.MeanMedianImputer(imputation_method = 'median', variables = continuous)),

    ('imputer_num_arbit',

     mdi.ArbitraryNumberImputer(arbitrary_number=-999,

                                variables=discrete)),

     ('cat impute',mdi.CategoricalVariableImputer(variables=categorical)),

    ('one hot', ce.OneHotCategoricalEncoder(top_categories= None, variables = categorical)),  

    

])
from sklearn.model_selection import train_test_split
X = a.drop(['SalePrice','Id'],axis=1)

y = np.log(a['SalePrice'])

print(len(X[:]),len(y))

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state = 44)

X_train.dtypes
x_train = pipe.fit_transform(X_train)

x_test= pipe.transform(X_test)

c = b.drop(['Id'],axis = 1)

c1 = pipe.transform(c)
import scipy

import matplotlib.pyplot as plt

from scipy.cluster import hierarchy as hc

corr = np.round(scipy.stats.spearmanr(x_train).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(30,150))

dendrogram = hc.dendrogram(z, labels=x_train.columns, orientation='left', leaf_font_size=20)

plt.show()
x_train.shape

x_test.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



rf = RandomForestRegressor(n_jobs = -1,n_estimators = 50, oob_score = True)



rf.fit(x_train,y_train)

rf.score(x_test,y_test)
#RMSE

from sklearn.metrics import mean_squared_error

import math

predicted = rf.predict(x_test)

mse = mean_squared_error(y_test, predicted)

rmse = math.sqrt(mse)

print(rmse)


importance = rf.feature_importances_

# plot feature importance

dic = {}



for i in range(len(importance)):

    dic[c1.columns[i]] = importance[i]

#most important feature

import matplotlib.pyplot as plt

from collections import Counter

k = Counter(dic)

dic1 = dict(k.most_common(5))

plt.bar(dic1.keys(),dic1.values())

plt.show()
%matplotlib inline

from pdpbox import pdp

def plot_pdp(feat_name, clusters=None):

#feat_name = feat_name or feat

    p = pdp.pdp_isolate(rf, x_train, feature=feat_name, model_features=x_train.columns)

    return pdp.pdp_plot(p, feat_name, plot_lines=True,

                   cluster=clusters is not None, n_cluster_centers=clusters)



l = ['OverallQual', 'GrLivArea', 'TotalBsmtSF','GarageArea','1stFlrSF']

for i in l:

     plot_pdp(feat_name = i)



impor = x_train[dic1.keys()]

impor.nunique()
test_data_labels = rf.predict(c1)



# Create predictions to be submitted!

pd.DataFrame({'Id': b.Id, 'SalePrice': np.exp(test_data_labels)}).to_csv('solution_base.csv', index =False)  

print("Done :D")