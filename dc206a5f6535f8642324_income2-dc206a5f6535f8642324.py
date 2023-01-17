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
train = pd.read_csv('/kaggle/input/tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv') 

test = pd.read_csv('/kaggle/input/tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv')



# train = train.drop(['Wears Glasses', 'Hair Color', 'Body Height [cm]'], axis=1)

# test = test.drop(['Wears Glasses', 'Hair Color', 'Body Height [cm]'], axis=1)
train.head()
# # when BACHELOR income is more than 5000000 

# train = train[train['Income in EUR'] < 5000000 ]



# # when AGE is 103 and income is more than 3000000(outlier in plot)

# train = train[train['Instance'] != 54704 ]

# # when AGE is more than 112, no more relevant data

# train = train[train['Age'] < 112]
all_data = pd.concat((train.loc[:,'Year of Record':'University Degree'],

                      test.loc[:,'Year of Record':'University Degree']))

all_data.columns
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

income = pd.DataFrame({"Income":train["Income in EUR"], "log(Income + 1)":np.log1p(train["Income in EUR"])})

income.hist()
#log transform the target:

train["Income in EUR"] = np.log1p(train["Income in EUR"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
train['Gender'] = train['Gender'].fillna('UNKNOWN')

train['Gender'] = train['Gender'].replace('0', 'UNKNOWN')

train['Gender'] = train['Gender'].replace('unknown', 'UNKNOWN')



test['Gender'] = test['Gender'].fillna('UNKNOWN')

test['Gender'] = test['Gender'].replace('0', 'UNKNOWN')

test['Gender'] = test['Gender'].replace('unknown', 'UNKNOWN')





train['University Degree'] = train['University Degree'].replace('0', 'Unknown University')

train['University Degree'] = train['University Degree'].fillna('Unknown University')



test['University Degree'] = test['University Degree'].replace('0', 'Unknown University')

test['University Degree'] = test['University Degree'].fillna('Unknown University')





train['Profession'] = train['Profession'].fillna('Unknown Profession')

test['Profession'] = test['Profession'].fillna('Unknown Profession')



train['Income in EUR'] = train['Income in EUR'].fillna(int(train['Income in EUR'].mean()))



all_data = pd.get_dummies(all_data)





# # Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

# plt.scatter(train['University Degree'], train['Income in EUR'], c = "blue", marker = "s")

# plt.title("Looking for outliers")

# plt.xlabel("University Degree")

# plt.ylabel("Income in EUR")

# plt.show()

len(train)


# dum = train[train['Income in EUR'] < 5000000]

# len(dum)
# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

# plt.scatter(train['Gender'], train['Income in EUR'], c = "blue", marker = "s")

# plt.title("Looking for outliers")

# plt.xlabel("Gender")

# plt.ylabel("Income in EUR")

# plt.show()

# dum = train[train['Income in EUR'] > 3000000]

# dum
# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

# plt.scatter(train['Age'], train['Income in EUR'], c = "blue", marker = "s")

# plt.title("Looking for outliers")

# plt.xlabel("Age")

# plt.ylabel("Income in EUR")

# plt.show()

# dum = train[train['Age'] > 100]

# dum

# # Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

# plt.scatter(dum['Age'], dum['Income in EUR'], c = "blue", marker = "s")

# plt.title("Looking for outliers")

# plt.xlabel("Age more than 100")

# plt.ylabel("Income in EUR")

# plt.show()

# len(dum)
# du = train[train['Instance'] != 54704]

# du
# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

# plt.scatter(train['Gender'], train['Income in EUR'], c = "blue", marker = "s")

# plt.title("Looking for outliers")

# plt.xlabel("Gender")

# plt.ylabel("Income in EUR")

# plt.show()
# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

# plt.scatter(train['Year of Record'], train['Income in EUR'], c = "blue", marker = "s")

# plt.title("Looking for outliers")

# plt.xlabel("Year of Record")

# plt.ylabel("Income in EUR")

# plt.show()
# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

# plt.scatter(train['Profession'], train['Income in EUR'], c = "blue", marker = "s")

# plt.title("Looking for outliers")

# plt.xlabel("Profession")

# plt.ylabel("Income in EUR")

# plt.show()
# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

# plt.scatter(train['Country'], train['Income in EUR'], c = "blue", marker = "s")

# plt.title("Looking for outliers")

# plt.xlabel("Country")

# plt.ylabel("Income in EUR")

# plt.show()
# print("Find most important features relative to target")

# train.columns

# corr = all_data.corr()

# corr.sort_values(["Income in EUR"], ascending = False, inplace = True)

# print(corr["Income in EUR"])
#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]



X_test = all_data[train.shape[0]:]

y = train['Income in EUR']

X_train.isnull().sum().sum()

y.isnull().sum()
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_ridge = Ridge()
# alphas = [0.05, 0.1, 0.3, 1, 2, 2.5, 3, 3.5, 4, 5]

# cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

#             for alpha in alphas]



# cv_ridge = pd.Series(cv_ridge, index = alphas)

# cv_ridge.plot(title = "Validation - Just Do It")

# plt.xlabel("alpha")

# plt.ylabel("rmse")
# cv_ridge.min()
# model_cv_ridge = RidgeCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
# model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
# rmse_cv(model_lasso).mean()
# coef = pd.Series(model_lasso.coef_, index = X_train.columns)

# print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])

# matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

# imp_coef.plot(kind = "barh")

# plt.title("Coefficients in the Lasso Model")

# #let's look at the residuals as well:

# matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



# preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

# preds["residuals"] = preds["true"] - preds["preds"]

# preds.plot(x = "preds", y = "residuals",kind = "scatter")
# from sklearn.linear_model import LinearRegression



# model_linear = LinearRegression().fit(X_train, y)

# from sklearn.linear_model import SGDRegressor

# from sklearn.metrics import mean_squared_error, r2_score

# sgd_model = SGDRegressor().fit(X_train, y)
from keras.models import Sequential

from keras.optimizers import Adam

from keras.layers.core import Dense

from keras.models import Model

from keras.layers.core import Activation

from keras.layers import Dropout



def create_model(dim, regress=False):

    model = Sequential()

    model.add(Dropout(0.2,input_shape=(dim,)))

    model.add(Dense(64, input_dim=dim, activation="relu"))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(24, activation="relu"))

    model.add(Dense(24, activation="relu"))

    model.add(Dense(16, activation="relu"))

    model.add(Dense(8, activation="relu"))

    model.add(Dense(1, activation="linear"))

    return model
model = create_model(X_train.shape[1], regress=True)

opt = Adam(lr=1e-2, decay=1e-3 / 200)

model.compile(loss="mean_squared_error", optimizer=opt)



# train the model

print("training model...")

model.fit(X_train, y, epochs=30, batch_size=100)
# import xgboost as xgb
# dtrain = xgb.DMatrix(X_train, label = y)

# dtest = xgb.DMatrix(X_test)



# params = {"max_depth":2, "eta":0.1}

# model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
# model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
# model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

# model_xgb.fit(X_train, y)
# xgb_preds = np.expm1(model_xgb.predict(X_test))

# cv_ridge_preds = np.expm1(model_cv_ridge.predict(X_test))

# lasso_preds = np.expm1(model_lasso.predict(X_test))

# linear_preds = np.expm1(model_linear.predict(X_test))

# sgd_preds = np.expm1(sgd_model.predict(X_test))

cnn_preds = np.expm1(model.predict(X_test))

# predictions = pd.DataFrame({"cvRidge":cv_ridge_preds, "lasso":lasso_preds})

# predictions.plot(x = "cvRidge", y = "lasso", kind = "scatter")
# preds = 0.7*lasso_preds + 0.3*cv_ridge_preds

# preds = 1.0*cv_ridge_preds

# preds = 1.0*linear_preds

# preds = 1.0*sgd_preds

preds = 1.0*cnn_preds
preds = preds[:,0]
preds.shape
solution = pd.DataFrame({"Instance":test.Instance, "Income":preds})

solution

solution.to_csv("tcd ml 2019-20 income prediction submission file.csv", index = False)