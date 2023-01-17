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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

import xgboost as xgb



%config InlineBackend.figure_format = 'retina' 

%matplotlib inline
train_set = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', header=0)

test_set =pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', header=0)
train_set.head()
train_set.info()
train_set.describe()
full_data = pd.concat((train_set.loc[:,'MSSubClass':'YrSold'],test_set.loc[:,'MSSubClass':'YrSold']))
matplotlib.rcParams['figure.figsize'] = (20, 6)

prices = pd.DataFrame({"price":train_set["SalePrice"], "log(price + 1)":np.log1p(train_set["SalePrice"])})

prices.hist()
#log transform:

train_set["SalePrice"] = np.log1p(train_set["SalePrice"])



#log transform skewed numeric features:

numeric_feats = full_data.dtypes[full_data.dtypes != "object"].index



skewed_feats = train_set[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



full_data[skewed_feats] = np.log1p(full_data[skewed_feats])
full_data = pd.get_dummies(full_data) # Converting categorical variable into dummy/indicator variables.
full_data = full_data.fillna(full_data.mean()) # Filling NA's with the mean of the column.
full_data #just checking
#creating matrices for sklearn:

x_train = full_data[:train_set.shape[0]]

x_test = full_data[train_set.shape[0]:]

y = train_set.SalePrice
def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, x_train, y, scoring="neg_mean_squared_error", cv = 5)) #simple cross-validation

    return(rmse)
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(x_train, y)
r = rmse_cv(model_lasso).mean()

print(r)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

print(cv_ridge.min())
matplotlib.rcParams['figure.figsize'] = (20, 6)

cv_ridge.plot(title = "Change of Error", color="r")

plt.xlabel("alpha")

plt.ylabel("rmse")
coef = pd.Series(model_lasso.coef_, index = x_train.columns)



print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])



matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)

imp_coef.plot(kind = 'barh',color='g')

plt.title("Coefficients in the Lasso Model")
plt.scatter(train_set.GrLivArea, train_set.SalePrice, c = "c",)

plt.title("GrLivArea vs SalePrice", size=20)

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.show()
#removing outliers

train_set = train_set[train_set.GrLivArea < 8.25]

train_set = train_set[train_set.SalePrice < 13]

train_set = train_set[train_set.SalePrice > 10.75]

train_set.drop("Id", axis=1, inplace=True)
matplotlib.rcParams['figure.figsize'] = (25.0, 15.0)



preds = pd.DataFrame({"preds":model_lasso.predict(x_train), "true":y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter", color="m")
dtrain = xgb.DMatrix(x_train, label = y)

dtest = xgb.DMatrix(x_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
matplotlib.rcParams['figure.figsize'] = (20, 8)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(x_train, y)
xgb_preds = np.expm1(model_xgb.predict(x_test))

lasso_preds = np.expm1(model_lasso.predict(x_test))



predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
preds = (0.7*lasso_preds) + (0.3*xgb_preds)
solution = pd.DataFrame({"id":test_set.Id, "SalePrice":preds})

solution.to_csv("housePrice_solution.csv", index = False)