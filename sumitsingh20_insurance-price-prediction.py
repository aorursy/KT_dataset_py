import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from sklearn.linear_model import LinearRegression
df = pd.read_csv('../input/insurance/insurance.csv')

df.head()
df.info()
df.describe()
df.isna().sum()
sns.pairplot(df)
corr = df.corr()

corr
fig,ax = plt.subplots(figsize = (8,8))

ax = sns.heatmap(corr,

                 annot = True,

                 linewidths = 0.5,

                 fmt = '.2f',

                 cmap = 'YlGnBu');
df.head()
fig, ax = plt.subplots(figsize=(6,6))

sns.barplot(x = "sex", y = "charges", ax=ax, data=df);
fig, ax = plt.subplots(figsize=(10,6))

sns.barplot(x = "children", y = "charges", ax=ax, data=df);
fig, ax = plt.subplots(figsize=(6,6))

sns.barplot(x = "smoker", y = "charges", ax=ax, data=df);
fig, ax = plt.subplots(figsize=(8,6))

sns.barplot(x = "region", y = "charges", ax=ax, data=df);
fig, ax = plt.subplots(figsize=(16,8))

sns.scatterplot(x = "bmi", y = "charges", ax=ax, data=df);
fig, ax = plt.subplots(figsize=(16,8))

sns.scatterplot(x = "age", y = "charges", ax=ax, data=df);
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['sex'] = le.fit_transform(df['sex'])

df['smoker'] = le.fit_transform(df['smoker'])

df['region'] = le.fit_transform(df['region'])
df.head()
df.info()
x = df.drop('charges',axis = 1)

y = df['charges']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
%%time

model = RandomForestRegressor(n_estimators = 1000,random_state = 42)

model.fit(x_train,y_train)
print(model.score(x_train,y_train))

print(model.score(x_test,y_test))
rf_preds = model.predict(x_test)
preds = pd.DataFrame({'Actual Charges': y_test,

                      'Predicted Charges': rf_preds,

                      'Difference': rf_preds - y_test})

preds.head()
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)

feat_importances.nlargest(5).plot(kind='barh')

plt.title('RandomForest Feature impotance')

plt.show()
xg_model = xgb.XGBRegressor()

xg_model.fit(x_train,y_train)
print(xg_model.score(x_train,y_train))

print(xg_model.score(x_test,y_test))

xg_preds = xg_model.predict(x_test)
xpreds = pd.DataFrame({'Actual Charges': y_test,

                      'Predicted Charges': xg_preds,

                      'Difference': xg_preds - y_test})

xpreds.head()
print(xg_model.feature_importances_)
xfeat_importances = pd.Series(xg_model.feature_importances_, index=x.columns)

xfeat_importances.nlargest(5).plot(kind='barh')

plt.title('XGBoost Feature impotance')

plt.show()
reg = LinearRegression()

reg.fit(x_train,y_train)
print(reg.score(x_train,y_train))

print(reg.score(x_test,y_test))
reg_preds = reg.predict(x_test)
rpreds = pd.DataFrame({'Actual Charges': y_test,

                      'Predicted Charges': reg_preds,

                      'Difference': reg_preds - y_test})

rpreds.head()
scores = pd.DataFrame({'RandomForest': model.score(x_test,y_test),

                       'XGBoost': xg_model.score(x_test,y_test),

                       'LinearRegression': reg.score(x_test,y_test)},

                      index = [0])

scores
scores.T.plot(kind = 'bar',

              figsize = (10,10))

plt.title('Scores of all Model')

plt.xlabel('Model Name')

plt.ylabel('Scores');