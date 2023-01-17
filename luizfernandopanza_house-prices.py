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
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.shape
train.head()
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], 
                      test.loc[:,'MSSubClass':'SaleCondition']))
all_data.head(10)
all_data.columns
all_data.dtypes
new_price = {"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])}
prices = pd.DataFrame(new_price)
mpl.rcParams['figure.figsize'] = (20, 10)
prices.hist()
train["SalePrice"] = np.log1p(train["SalePrice"])
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) 
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
skewed_feats
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y_train = train.SalePrice
def rmse_cv(modelo):
    rmse = np.sqrt(-cross_val_score(modelo, 
                                    X_train, 
                                    y_train, 
                                    scoring = "neg_mean_squared_error", 
                                    cv = 5))
    return(rmse)
modelo_lr = LinearRegression(normalize = False, fit_intercept = True)
modelo_lr.fit(X_train, y_train)
error1 = (rmse_cv(modelo_lr).mean()) *100
print('This is error 1, linear Regression with no regularization: %0.2f' % error1 + '% error rate')
modelo_ridge = Ridge()
cross_val_score(modelo_ridge, 
                X_train, 
                y_train, 
                scoring = "neg_mean_squared_error", 
                cv = 5)
rmse_ridge = np.sqrt(-cross_val_score(modelo_ridge, 
                                      X_train, 
                                      y_train, 
                                      scoring = "neg_mean_squared_error", 
                                      cv = 5))
rmse_ridge
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validação")
plt.xlabel("Alpha")
plt.ylabel("RMSE")
error2 = (cv_ridge.mean()) *100
print('This is error 2, linear Regression RIDGE: %0.2f' % error2 + '% error rate')
modelo_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)
error3 = (rmse_cv(modelo_lasso).mean()) *100
print('This is error 3, linear Regression LASSO: %0.2f' % error3 + '% error rate')
coef = pd.Series(modelo_lasso.coef_, index = X_train.columns)
coef.head()
imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
mpl.rcParams['figure.figsize'] = (20, 10)
imp_coef.plot(kind = "barh")
plt.title("Coeficientes no Modelo LASSO")