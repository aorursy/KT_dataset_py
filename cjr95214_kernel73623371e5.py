# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
# all_data = pd.concat([train_data.loc[:,'MSSubClass':'SaleCondition'],test_data.loc[:,'MSSubClass':'SaleCondition']])

# all_data.head()
#log transform

train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
total = train_data.isnull().sum().sort_values(ascending=False)

percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
train_data = train_data.drop((missing_data[missing_data['Total'] > 1]).index,1)

train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
train_data.shape
total2 = test_data.isnull().sum().sort_values(ascending=False)

percent2 = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending=False)

missing_data2 = pd.concat([total2, percent2], axis=1, keys=['Total2', 'Percent2'])

missing_data2.head(34)
test_data = test_data.drop((missing_data[missing_data['Total'] > 1]).index,1)

total2 = test_data.isnull().sum().sort_values(ascending=False)

percent2 = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending=False)

missing_data2 = pd.concat([total2, percent2], axis=1, keys=['Total2', 'Percent2'])

missing_data2.head(34)
test_data.shape
all_data = pd.concat([train_data.loc[:,'MSSubClass':'SaleCondition'],test_data.loc[:,'MSSubClass':'SaleCondition']])

all_data.head()
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index #find all the numeric features

numeric_feats
skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness 给每个数值型属性计算偏度
#找出偏度大于0.75的属性

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index 

#对train+test合并数据的偏度大于0.75的属性进行log转换

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data) #one-hot
all_data = all_data.fillna(all_data.mean())

all_data.isnull().sum()
X_train = all_data[:train_data.shape[0]]

X_test = all_data[train_data.shape[0]:]

y_train = train_data.SalePrice
from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 8 ))

    return(rmse)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

ridge_model = RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75],cv=10)

ridge_model.fit(X_train,y_train)
rmse_cv(ridge_model).mean()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005],cv=8).fit(X_train, y_train)

rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_ ,  index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
lasso_preds = np.expm1(model_lasso.predict(X_test))
ridge_preds = np.expm1(ridge_model.predict(X_test))
# from sklearn.ensemble import RandomForestRegressor

# from sklearn.model_selection import RandomizedSearchCV

# parameters = {'max_features':[0.6,0.7,0.8,0.75]}

# model = RandomizedSearchCV(estimator=RandomForestRegressor(n_estimators=153,max_depth=11,n_jobs=-1,criterion='mse',random_state=33),param_distributions=parameters,cv=6,n_jobs=-1,scoring='neg_mean_squared_error')

# model.fit(X_train, y_train)

# print(model.best_score_)

# print(model.best_params_)



#-0.02034703234067721                               -0.019688878072480036

#{'n_estimators': 155, 'max_depth': 11}        {'n_estimators': 153, 'max_depth': 11}
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=153,max_depth=11,n_jobs=-1,criterion='mse',random_state=33,max_features=0.63)

rf_model.fit(X_train, y_train)
rmse_cv(rf_model).mean()
rf_preds = np.expm1(rf_model.predict(X_test))
rf_preds
ridge_preds
lasso_preds
result= 0.5*lasso_preds +0.3*ridge_preds + 0.2*rf_preds
result
submission = pd.DataFrame({"id":test_data.Id, "SalePrice":result})

submission.to_csv("submission.csv", index = False)