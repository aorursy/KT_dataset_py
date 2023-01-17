import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy.stats import skew
X_train = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')
X_train.head()
X_train.info()
#the correlation matrix

corr_mat = X_train.corr()

corr_mat['SalePrice'].sort_values(ascending=False)

#get the variables that have a good correlation(>.7) with the target

values = list(corr_mat['SalePrice'].values)

keys = list(corr_mat['SalePrice'].keys())

variables = [i for i in keys if values[keys.index(i)] > .7]



#ploting scatters 

sns.set()

sns.pairplot(X_train[variables],size=3)

plt.show()
X_train.drop(X_train[(X_train['GrLivArea']>4000) & (X_train['SalePrice']<300000)].index, inplace=True)
sns.distplot(X_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(X_train['SalePrice'], plot=plt)

plt.show()
#saving the labels of our target variable 

y = X_train['SalePrice']



#save the id column for later.

test_id = X_test['Id']



#dropping column that will not be needed when training our model.

X_train.drop('SalePrice', 1, inplace=True)

X_train.drop("Id", axis = 1, inplace = True)

X_test.drop("Id", axis = 1, inplace = True)



#save indexes before concatenating

X_train_idx = X_train.shape[0]

X_test_idx = X_test.shape[0]



#concatenate the data

data = pd.concat((X_train,X_test)).reset_index(drop=True)

print('Train index: %s \nTest index: %s \nShape of our whole data(rows,columns): %s'%(X_train_idx, X_test_idx, data.shape))
#print the sum of the missing values of a column from the highest to the lowest.

numeric_cols = []

non_numeric_cols = []

for key,val in data.isnull().sum().sort_values(ascending=False).items():

    if val > 0:

        print(key, val, data[key].dtype)

        if data[key].dtype != 'object':numeric_cols.append(key)

        elif data[key].dtype == 'object':non_numeric_cols.append(key)

        
features = data.isnull().sum().sort_values(ascending=False).keys()

missing_values = pd.DataFrame(data[numeric_cols])

missing_values.head()
fill_zero_cols = ['BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF2', 'GarageCars']

fill_median_cols = ['GarageArea','TotalBsmtSF', 'MasVnrArea', 'BsmtFinSF1', 'LotFrontage', 'BsmtUnfSF', 'GarageYrBlt']

[data[i].fillna(0,inplace=True) for i in fill_zero_cols]

[data[j].fillna(data[j].median(),inplace=True) for j in fill_median_cols];
features = data.isnull().sum().sort_values(ascending=False).keys()

missing_values = pd.DataFrame(data[non_numeric_cols])

missing_values.head()
data.fillna('none', inplace=True)
data.isnull().sum().sort_values(ascending=False)
#transforming our target labels(SalePrice).

y = np.log1p(y)

#transfomring the numeric features 

numeric_feats = data.dtypes[data.dtypes != "object"].index

skewed_feats = X_train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

data[skewed_feats] = np.log1p(data[skewed_feats])
pd.get_dummies(data['PoolQC']).head()
data = pd.get_dummies(data)
X_train = data[:X_train_idx]

X_test = data[X_train_idx:] 
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
#our cross function

def display_scores(scores):

    print("Score:{:.4f}".format(scores.mean()))

#learning curves function 

def plot_learning_curves(model, X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):

        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m])

        y_val_predict = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))

        val_errors.append(mean_squared_error(y_val_predict, y_val))

        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")

        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
elastic_net = ElasticNet(alpha=.5, l1_ratio=.5)

scores = cross_val_score(elastic_net, X_train, y, scoring='neg_mean_squared_error', cv=5)

rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)
from sklearn.model_selection import GridSearchCV

param_grid = [

        {'alpha': [.0004, .0005], 'l1_ratio': [.5, .8, 1]},

]

    

grid_search = GridSearchCV(ElasticNet(), param_grid, cv=5,

                               scoring='neg_mean_squared_error',

                               return_train_score=True)
grid_search.fit(X_train, y)
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
lasso = Lasso(alpha=.1).fit(X_train,y)

scores = cross_val_score(lasso,X_train,y,scoring='neg_mean_squared_error',cv=5)

rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)
plot_learning_curves(lasso,X_train,y)

plt.show()
lasso = Lasso(alpha=.0005).fit(X_train,y)

scores = cross_val_score(lasso,X_train,y,scoring='neg_mean_squared_error',cv=5)

rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)
plot_learning_curves(lasso,X_train,y)

plt.show()
preds = lasso.predict(X_test)

lasso_predictions = pd.DataFrame(np.exp(preds), test_id, columns=['SalePrice']).to_csv('lasso_predictions.csv')