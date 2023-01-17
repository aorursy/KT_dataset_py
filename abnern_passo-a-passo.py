import numpy as np

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sbn

from sklearn.linear_model import Ridge, Lasso, LassoCV

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from scipy.stats import skew



print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
train.describe()
print(train.dtypes)
print('Training set size:', len(train))

train_after_na = train.dropna()

print('Training set size after removing na:', len(train_after_na))



plt.rcParams['figure.figsize'] = (20.0, 6.0)

train.isnull().sum().plot.bar()
for col_name in train.columns[train.isnull().any()].tolist():

    print(col_name, train[col_name].isnull().sum())
df = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))
df = pd.get_dummies(df)

df.head()
df = df.fillna(df.mean()) #substituindo os valores na pela média
#agora dividindo de novo em training e test:

X_train = df[:train.shape[0]]

X_test = df[train.shape[0]:]

y = np.log1p(train.SalePrice) # na descrição do projeto fala pra usar log
alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50]

rmse = []

for a in alphas:

    error = np.sqrt(-cross_val_score(Ridge(alpha=a), X_train, y, scoring="neg_mean_squared_error", cv = 5))

    rmse.append( error.mean() )
plt.rcParams['figure.figsize'] = (10.0, 6.0)

plt.plot(alphas, rmse)
print(np.mean(rmse))

print(np.mean(y), np.std(y))
from yellowbrick.regressor import ResidualsPlot

model1 = Ridge(alpha=10)

visualizer = ResidualsPlot(model1)



visualizer.fit(X_train, y)

visualizer.score(X_train, y)

visualizer.poof() 
model2 = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], max_iter=20000, cv=5).fit(X_train, y)
error = np.sqrt(-cross_val_score(model2, X_train, y, scoring="neg_mean_squared_error", cv = 5))

print(error.mean() )
plt.rcParams['figure.figsize'] = (40.0, 6.0)

coef = pd.Series(model2.coef_, index = X_train.columns)

coef.sort_values(inplace = True,  ascending=False)

coef.plot(kind = "bar")
coef.head(10)
print(sum(coef != 0), len(X_train.columns))
plt.rcParams['figure.figsize'] = (10.0, 6.0)

visualizer = ResidualsPlot(model2)



visualizer.fit(X_train, y)

visualizer.score(X_train, y)

visualizer.poof() 
import xgboost as xgb
param = {"max_depth":4, "eta":0.1}

dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)

mdcv = xgb.cv(param, dtrain,  num_boost_round=500, early_stopping_rounds=100)
plt.rcParams['figure.figsize'] = (10.0, 6.0)

mdcv[['train-rmse-mean','test-rmse-mean'] ].plot()
plt.rcParams['figure.figsize'] = (10.0, 6.0)

mdcv[['train-rmse-std','test-rmse-std'] ].plot()
model3 = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.1)

model3.fit(X_train, y)
plt.rcParams['figure.figsize'] = (10.0, 20.0)

xgb.plot_importance(model3)
plt.rcParams['figure.figsize'] = (10.0, 6.0)

visualizer = ResidualsPlot(model3)



visualizer.fit(X_train, y)

visualizer.score(X_train, y)

visualizer.poof() 
error = np.sqrt(-cross_val_score(model3, X_train, y, scoring="neg_mean_squared_error", cv = 5))

print(error.mean() )
df2 = StandardScaler().fit_transform(df)



pca = PCA(0.95)

df2 = pca.fit_transform(df2)



X_train = df2[:train.shape[0]]

X_test = df2[train.shape[0]:]

y = np.log1p(train.SalePrice)

print('Antes do PCA: ', df.shape[1], 'Depois do PCA:', df2.shape[1])
model3_1 = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.1)

model3_1.fit(X_train, y)



error = np.sqrt(-cross_val_score(model3_1, X_train, y, scoring="neg_mean_squared_error", cv = 5))

print(error.mean() )
model2_1 = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], max_iter=20000, cv=5).fit(X_train, y)



error = np.sqrt(-cross_val_score(model2_1, X_train, y, scoring="neg_mean_squared_error", cv = 5))

print(error.mean() )
model1_1 = Ridge(alpha=10)

model1_1.fit(X_train, y)

error = np.sqrt(-cross_val_score(model1_1, X_train, y, scoring="neg_mean_squared_error", cv = 5))

print(error.mean() )
#eu devia ter feito uma função de parte desse codigo pra nao ter que colar tudo de novo aqui

df = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))

numeric_feats = df.dtypes[df.dtypes != "object"].index

theshold = 0.75

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))

skewed_feats = skewed_feats[skewed_feats > theshold]

skewed_feats = skewed_feats.index



df[skewed_feats] = np.log1p(df[skewed_feats])

df = df.fillna(df.mean())
df = df.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4}}

                     )



df["Qual3"] = df.OverallQual.replace({1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:3})  

df["Cond3"] = df.OverallCond.replace({1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:3})

df["Kitchen5"] = df.KitchenQual.replace({1:1, 2:1, 3:1, 4:2, 5:2 })

df["Extern5"] = df.ExterCond.replace({1:1, 2:1, 3:1, 4:2, 5:2 })



df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]

df["GarageGrade"] = df["GarageQual"] * df["GarageCond"]

df["ExterGrade"] = df["ExterQual"] * df["ExterCond"]

df["KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"]

df["Grade3"] = df["Qual3"] * df["Cond3"]
df = pd.get_dummies(df)

df = df.fillna(df.mean())



X_train = df[:train.shape[0]]

X_test = df[train.shape[0]:]

y = np.log1p(train.SalePrice)
df.head()
model2_2 = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], max_iter=40000, cv=5).fit(X_train, y)

error = np.sqrt(-cross_val_score(model2_2, X_train, y, scoring="neg_mean_squared_error", cv = 5))

print(error.mean() )
model1_2 = Ridge(alpha=10)

model1_2.fit(X_train, y)

error = np.sqrt(-cross_val_score(model1_2, X_train, y, scoring="neg_mean_squared_error", cv = 5))

print(error.mean() )
model3_2 = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.1)

model3_2.fit(X_train, y)



error = np.sqrt(-cross_val_score(model3_2, X_train, y, scoring="neg_mean_squared_error", cv = 5))

print(error.mean() )
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf',  gamma='scale')



error = np.sqrt(-cross_val_score(svr_rbf, X_train, y, scoring="neg_mean_squared_error", cv = 5, n_jobs=4))

print(error.mean() )
lasso_y = np.expm1(model2_2.predict(X_test))

submission_df = pd.DataFrame({"id":test.Id, "SalePrice":lasso_y})

submission_df.to_csv("lasso_abner.csv", index = False)