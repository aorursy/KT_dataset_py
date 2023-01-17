import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
df = pd.read_csv('../input/boston-housing-price/housing_price.csv')
df.head()
df.shape
df = df.fillna(df.mean())
X = pd.DataFrame(df.iloc[:, 0:13])
y = pd.DataFrame(df.iloc[:, 13:14])
scaling = StandardScaler()
X_std = scaling.fit_transform(df.iloc[:, 0:13])
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.2, random_state = 1)
lr1 = LinearRegression()
model1 = lr1.fit(X_train, y_train)
coefs1 = dict(zip(X.columns, abs(lr1.coef_[0])))
coefs1
y_pred = lr1.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred)
mse1.round(4)
r2_score(y_test, y_pred).round(4)
lr1.score(X_test, y_test).round(4)
rfe = RFE(estimator = LinearRegression(), n_features_to_select = 11, verbose = 1)
rfe.fit(X_train, y_train)
X.columns[rfe.support_]
print(dict(zip(X.columns, rfe.ranking_)))
y1_pred = rfe.predict(X_test)
mse2 = mean_squared_error(y_test, y1_pred)
mse2.round(4)
r2_score(y_test, y1_pred).round(4)
rfe.score(X_test, y_test).round(4)