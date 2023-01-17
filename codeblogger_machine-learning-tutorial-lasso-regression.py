# import the necessary packages

import numpy as np

import pandas as pd

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import model_selection

from sklearn.linear_model import LassoCV

from scipy.stats import boxcox

import matplotlib.pyplot as plt
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



lasso_model = Lasso().fit(X_train,y_train)

lasso_model
print("intercept: ", lasso_model.intercept_)

print("coef: ", lasso_model.coef_)
# coefficients for different lambda values



alphas = 10**np.linspace(10, -2, 100) * 0.5

lasso = Lasso()

coefs = []



for a in alphas:

    lasso.set_params(alpha=a)

    lasso.fit(X_train,y_train)

    coefs.append(lasso.coef_)
ax = plt.gca()

ax.plot(alphas*2, coefs)

ax.set_xscale("log")

plt.axis("tight")

plt.xlabel("alpha")

plt.show()
lasso.predict(X_test)[0:10]
y_pred = lasso.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
lasso_cv_model = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)

lasso_cv_model
lasso_cv_model.fit(X_train, y_train)
lasso_cv_model.alpha_
lasso_tuned = Lasso().set_params(alpha= lasso_cv_model.alpha_).fit(X_train,y_train)

y_pred = lasso_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))