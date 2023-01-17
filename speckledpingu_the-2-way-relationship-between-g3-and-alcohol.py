# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

from pandas.tools.plotting import parallel_coordinates,andrews_curves

import numpy as np

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import scale

from sklearn.linear_model import LassoCV,RidgeCV,Lasso,Ridge

from sklearn.metrics import mean_squared_error

from sklearn.cross_validation import train_test_split

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_mat = pd.read_csv("../input/student-mat.csv")

df_mat.info()
df_mat[['Dalc','Walc','G1','G2','G3']].describe()
df_mat[['Dalc','Walc','G1','G2','G3','Medu','Fedu']].corr()
parallel_coordinates(df_mat[['Dalc','Walc','G1','G2','G3','Medu','Fedu','health','absences']],'Dalc')
andrews_curves(df_mat[['Dalc','Walc','G1','G2','G3','Medu','Fedu','health','absences','studytime','traveltime','goout','freetime','famrel']],'Dalc')
basic_ols = smf.ols(formula="G3 ~ G1 + G2",data=df_mat)

basic_ols.fit().summary()
df_mat_heavy_drinking = df_mat[df_mat['Dalc']>=3]

df_mat_light_drinking = df_mat[df_mat['Dalc']<3]
basic_hd_ols = smf.ols(formula="G3 ~ G1 + G2",data=df_mat_heavy_drinking)

basic_hd_ols.fit().summary()
basic_ld_ols = smf.ols(formula="G3 ~ G1 + G2",data=df_mat_light_drinking)

basic_ld_ols.fit().summary()
andrews_curves(df_mat_heavy_drinking[['Dalc','Walc','G1','G2','G3','Medu','Fedu','health','absences','studytime','traveltime','goout','freetime','famrel']],'Dalc')
parallel_coordinates(df_mat_heavy_drinking[['Dalc','Walc','G1','G2','G3','Medu','Fedu','health','absences']],'Dalc')
df_mat_heavy_drinking.corr()
df_regressor_mat_hd = df_mat_heavy_drinking.drop(['school','sex','age','address','famsize','Pstatus','Fjob','Mjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'],axis=1)

df_regressor_mat_hd = pd.DataFrame(scale(df_regressor_mat_hd),columns=['Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3'])

X_train, X_test, y_train, y_test = train_test_split(df_regressor_mat_hd.drop('G3',axis=1),df_regressor_mat_hd.G3,random_state=42)
alphas = 10**np.linspace(-4,4,150)
lasso_coefs = []

lasso_mse = []

lasso = Lasso()

for alpha in alphas:

    lasso.set_params(alpha=alpha)

    lasso.fit(X_train,y_train)

    lasso_coefs.append(lasso.coef_)

    lasso_mse.append(mean_squared_error(y_test,lasso.predict(X_test)))
lasso_cv = LassoCV(alphas=alphas)

lasso_cv.fit(X_train,y_train)
plt.plot(alphas,lasso_coefs)

plt.xscale("log")

plt.axvline(lasso_cv.alpha_,linestyle="dashed",color='g',alpha=0.8)

plt.xlim(0.001,1)
plt.plot(alphas,lasso_mse)

plt.xscale("log")

plt.axvline(lasso_cv.alpha_,alpha=0.8,linestyle="dashed")
lasso_coefficients = pd.Series(lasso_cv.coef_,index=['Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2'])

lasso_coefficients
ridge_coefs = []

ridge_mse = []

ridge = Ridge()

for alpha in alphas:

    ridge.set_params(alpha=alpha)

    ridge.fit(X_train,y_train)

    ridge_coefs.append(ridge.coef_)

    ridge_mse.append(mean_squared_error(y_test,ridge.predict(X_test)))
ridge_cv = RidgeCV(alphas=alphas)

ridge_cv.fit(X_train,y_train)
plt.plot(alphas,ridge_coefs)

plt.xscale("log")

plt.axvline(ridge_cv.alpha_,linestyle="dashed",color='g',alpha=0.8)

plt.plot(alphas,ridge_mse)

plt.xscale("log")

plt.axvline(ridge_cv.alpha_,alpha=0.8,linestyle="dashed")
ridge_coefficients = pd.Series(ridge_cv.coef_,index=['Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2'])

ridge_coefficients
X_train, X_test, y_train, y_test = train_test_split(df_regressor_mat_hd.drop('Dalc',axis=1),df_regressor_mat_hd.Dalc,random_state=42)
ridge_dalc_coefs = []

ridge_dalc_mse = []

ridge = Ridge()

for alpha in alphas:

    ridge.set_params(alpha=alpha)

    ridge.fit(X_train,y_train)

    ridge_dalc_coefs.append(ridge.coef_)

    ridge_dalc_mse.append(mean_squared_error(y_test,ridge.predict(X_test)))



ridge_cv = RidgeCV(alphas=alphas)

ridge_cv.fit(X_train,y_train)



plt.plot(alphas,ridge_dalc_coefs)

plt.xscale("log")

plt.axvline(ridge_cv.alpha_,linestyle="dashed",color='g',alpha=0.8)

plt.xlim(0.01,10000)
plt.plot(alphas,ridge_dalc_mse)

plt.xscale("log")

plt.axvline(ridge_cv.alpha_,alpha=0.8,linestyle="dashed")
ridge_coefficients = pd.Series(ridge_cv.coef_,index=['Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Walc','health','absences','G1','G2','G3'])

ridge_coefficients