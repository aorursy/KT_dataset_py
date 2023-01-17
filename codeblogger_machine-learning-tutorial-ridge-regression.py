# import the necessary packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split

from sklearn import model_selection

from scipy.stats import boxcox

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import RidgeCV
# load data

data = "../input/insurance/insurance.csv"

df = pd.read_csv(data)



# show data (6 row)

df.head(6)
df_encode = pd.get_dummies(data = df, columns = ['sex','smoker','region'])

df_encode.head()
# normalization

y_bc,lam, ci= boxcox(df_encode['charges'],alpha=0.05)

df_encode['charges'] = np.log(df_encode['charges'])



df_encode.head()
X = df_encode.drop("charges",axis=1)

y = df_encode["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



ridge_model = Ridge(alpha=0.1).fit(X_train, y_train)

ridge_model
ridge_model.coef_
ridge_model.intercept_
lambdas = 10**np.linspace(10,-2,100)*0.5 # Creates random numbers

ridge_model =  Ridge()

coefs = []



for i in lambdas:

    ridge_model.set_params(alpha=i)

    ridge_model.fit(X_train,y_train)

    coefs.append(ridge_model.coef_)

    

ax = plt.gca()

ax.plot(lambdas, coefs)

ax.set_xscale("log")
ridge_model = Ridge().fit(X_train,y_train)



y_pred = ridge_model.predict(X_train)



print("Predict: ", y_pred[0:10])

print("Real: ", y_train[0:10].values)
RMSE = np.mean(mean_squared_error(y_train,y_pred)) # rmse = square root of the mean of error squares

print("train error: ", RMSE)
Verified_RMSE = np.sqrt(np.mean(-cross_val_score(ridge_model, X_train, y_train, cv=20, scoring="neg_mean_squared_error")))

print("Verified_RMSE: ", Verified_RMSE)
ridge_model = Ridge(10).fit(X_train,y_train)

y_pred = ridge_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
ridge_model = Ridge(30).fit(X_train,y_train)

y_pred = ridge_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
ridge_model = Ridge(90).fit(X_train,y_train)

y_pred = ridge_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
lambdas1 = 10**np.linspace(10,-2,100)

lambdas2 = np.random.randint(0,10000,100)



ridgeCV = RidgeCV(alphas = lambdas1,scoring = "neg_mean_squared_error", cv=10, normalize=True)

ridgeCV.fit(X_train,y_train)
ridgeCV.alpha_
# final model

ridge_tuned = Ridge(alpha = ridgeCV.alpha_).fit(X_train,y_train)

y_pred = ridge_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))*100
# for lambdas2

ridgeCV = RidgeCV(alphas = lambdas2,scoring = "neg_mean_squared_error", cv=10, normalize=True)

ridgeCV.fit(X_train,y_train)

ridge_tuned = Ridge(alpha = ridgeCV.alpha_).fit(X_train,y_train)

y_pred = ridge_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))*100