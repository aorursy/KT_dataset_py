import numpy as np

import pandas as pd

df = pd.read_csv("/kaggle/input/lianjia/new.csv", encoding='iso-8859-1', low_memory = False)

df.head()
df.info()
df.describe()
df.corr()
df = df[['totalPrice', 'square', 'renovationCondition', 'communityAverage']]

df.head()
df = df.fillna(df.mean())

df.info()
X = df.drop('totalPrice', axis = 1)

y = df['totalPrice']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
from sklearn import metrics

from sklearn.model_selection import cross_val_score



results_df = pd.DataFrame()

columns = ["Model", "Cross Val Score", "MAE", "MSE", "RMSE", "R2"]



def evaluate(true, predicted):

    mae = metrics.mean_absolute_error(true, predicted)

    mse = metrics.mean_squared_error(true, predicted)

    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))

    r2_square = metrics.r2_score(true, predicted)

    return mae, mse, rmse, r2_square



def append_results(model_name, model, results_df, y_test, pred):

    results_append_df = pd.DataFrame(data=[[model_name, *evaluate(y_test, pred) , cross_val_score(model, X, y, cv=10).mean()]], columns=columns)

    results_df = results_df.append(results_append_df, ignore_index = True)

    return results_df
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression(normalize=True)

lin_reg.fit(X_train,y_train)



pred = lin_reg.predict(X_test)



results_df = append_results("Linear Regression",  LinearRegression(), results_df, y_test, pred)

results_df
from sklearn.linear_model import RANSACRegressor



model = RANSACRegressor()

model.fit(X_train, y_train)



pred = model.predict(X_test)

results_df = append_results("Robust Regression",  RANSACRegressor(), results_df, y_test, pred)

results_df
from sklearn.linear_model import Ridge



model = Ridge()

model.fit(X_train, y_train)

pred = model.predict(X_test)

results_df = append_results("Ridge Regression",  Ridge(), results_df, y_test, pred)

results_df
from sklearn.linear_model import Lasso



model = Lasso()

model.fit(X_train, y_train)

pred = model.predict(X_test)

results_df = append_results("Lasso Regression",  Lasso(), results_df, y_test, pred)

results_df
from sklearn.linear_model import ElasticNet



model = ElasticNet()

model.fit(X_train, y_train)

pred = model.predict(X_test)

results_df = append_results("ElasticNet Regression",  ElasticNet(), results_df, y_test, pred)

results_df