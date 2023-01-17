# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn import metrics

from scipy.stats import skew

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import cross_val_score



train = pd.read_csv('../input/train.csv', low_memory=False, dtype = {'SalePrice': float})

test = pd.read_csv('../input/test.csv', low_memory=False)

train=train.drop(train.loc[:,['Condition2','Utilities','RoofMatl','Heating','PoolQC','Fence','MiscFeature','Street','Alley']],1)

test=test.drop(test.loc[:,['Condition2','Utilities','RoofMatl','Heating','PoolQC','Fence','MiscFeature','Street','Alley']],1)



all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))

#all_data.add('Age', axis=1, level=None, fill_value=None)

all_data['YearRemodAdd']=2016-all_data['YearRemodAdd']



#all_data.drop('YearRemodAdd',axis=1,inplace=True)

#train.drop('YearRemodAdd',axis=1,inplace=True)

#all_data=all_data[all_data.isnull().any(axis=1)]

for c in all_data:

    if sum(all_data[c].isnull()) >= 600:

        all_data.drop(c, axis=1, inplace=True)

#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.67]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])





all_data = pd.get_dummies(all_data)



#filling NA's with the mean of the column:

all_data = all_data.fillna(  1 )





#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice



print (X_train.shape, X_test.shape, y.shape)

#print (all_data['YearRemodAdd'].head)



def rmse_cv(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))

    return(rmse)
def runAlgo(algo, x, y, X_test):

    algo.fit(x, y)

    y_pred = algo.predict(X_test)

    print (np.sqrt(metrics.mean_squared_error(y, y_pred)))
#from sklearn.ensemble import RandomForestRegressor

# Algo 3

#rfr = RandomForestRegressor(n_estimators=800)

#rfr = RandomForestRegressor()

#rfr.fit(X_train, y)

#y_pred = np.expm1(rfr.predict(X_train))

#print (np.sqrt(metrics.mean_squared_error(y, y_pred)))



# Test data

#y_pred_ = np.expm1(rfr.predict(X_test))

#solution = pd.DataFrame({"Id":test.Id, "SalePrice":y_pred_})

#solution.to_csv("kaggle.csv", index = False)
#from sklearn.svm import SVR

#clf = SVR()

#clf.fit(X_train, y) 

#SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',

#    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

#NuSVR(C=1.0, cache_size=200, coef0=0.0, degree=3, gamma='auto',

     # kernel='rbf', max_iter=-1, nu=0.1, shrinking=True, tol=0.001,

     # verbose=False)



#y_pred_ = np.expm1(clf.predict(X_test))

#solution = pd.DataFrame({"Id":test.Id, "SalePrice":y_pred_})

#solution.to_csv("kaggle.csv", index = False)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)

model_lasso = LassoCV(alphas = [1.2, 0.27, 0.028, 0.0040,0.00043356]).fit(X_train, y)

preds = np.expm1(model_lasso.predict(X_test))

solution = pd.DataFrame({"Id":test.Id, "SalePrice":preds})

solution.to_csv("kaggle.csv", index = False)