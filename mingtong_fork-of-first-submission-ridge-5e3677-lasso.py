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
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.shape, test.shape
#train.describe()

#train.info()
#train["SalePrice"].hist();
#prices = np.log1p(train["SalePrice"])

#prices.hist()
train["SalePrice"] = np.log1p(train["SalePrice"])

y_train = train["SalePrice"]
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
from scipy.stats import skew

from scipy.stats.stats import pearsonr

#log transform skewed numeric features:

numeric_feats = train.dtypes[train.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index
'SalePrice' in skewed_feats
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

#Convert categorical variable into dummy/indicator variables

all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())

all_data.shape
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

X_train.shape, X_test.shape
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.4, random_state=0)

#X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.model_selection import cross_val_score

from sklearn import linear_model

from sklearn.metrics import mean_squared_error



#reg = linear_model.Ridge(alpha = .5)

#reg.fit(X_train, y_train) 

#y_pred = reg.predict(X_test)

#mean_squared_error(y_test, y_pred)  
#mean_squared_error(y_test, y_pred, multioutput='raw_values')
reg = linear_model.LassoCV(alphas = [0.0001, 0.0005, 0.001, 0.005])

reg.fit(X_train, y_train)

reg.alpha_   
#reg = linear_model.RidgeCV(alphas = [2.0, 5., 10, 20., 50.])

#reg.fit(X_train, y_train)

#reg.alpha_   
#reg = linear_model.RidgeCV(alphas = [6., 8., 10., 12., 14., 17.], fit_intercept=True, \

                           #normalize=False, scoring=None, cv=8, gcv_mode=None, store_cv_values=False)

#reg.fit(X_train, y_train)
print(reg.alpha_)

y_pred = reg.predict(X_test)

y_pred = np.expm1(y_pred)
output = pd.DataFrame(test['Id'])
output['SalePrice'] = y_pred
output
output.to_csv('LASSO0319.csv', index=False)