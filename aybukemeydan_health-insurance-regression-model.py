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
df = pd.read_csv("/kaggle/input/health-insurance-cost-prediction/insurance.csv")
df.head()
df.describe()
df.info() 
df.smoker.unique()
df.region.unique()
df.sex.unique()
import seaborn as sns
sns.lmplot(x="bmi",y="charges",hue="sex",data=df)
sns.lmplot(x="children",y="charges",hue="sex",data=df)
plt.figure(figsize=(8,6))
ax=sns.heatmap(df.corr(),annot=True,vmin=-1,vmax=1)
sns.pairplot(df) #lets check any linear connection btw features and target values
df.columns
import statsmodels.formula.api as smf
# Sadece nümerik değişkenlerle model açıklanamıyor.
model=smf.ols('charges~ age + bmi + children',data=df)
model.fit().summary() #underfitting ! UPS!
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.concat([df, pd.get_dummies(df["sex"],prefix="sex")], axis=1)
df = pd.concat([df, pd.get_dummies(df["region"],prefix="region")], axis=1)
df = pd.concat([df, pd.get_dummies(df["smoker"],prefix="smoker")], axis=1)
df.head()
df.drop([ "region", "smoker", "sex"], axis = 1, inplace = True)
plt.figure(figsize=(8,6))
ax=sns.heatmap(df.corr(),annot=True,vmin=-1,vmax=1)
df.drop(columns=["smoker_yes","sex_female"],inplace=True) # Avoid from multicol.
model=smf.ols('charges~ age + bmi + children + sex_male + region_northeast + region_northwest + region_southeast + region_southwest + smoker_no',data=df)
model.fit().summary() # More data reduces bias.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge,Lasso,RidgeCV,LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

X, y = df.drop('charges',axis=1), df['charges']

#Simple Validation
X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=3) # 60-20-20
lreg = LinearRegression()
lreg.fit(X_train,y_train)

pred = lreg.predict(X_val)

#mean square error
mse = np.mean((pred - y_val)**2)

#r2

lreg.score(X_val, y_val)
mse
coefs=pd.DataFrame(X_train.columns)
coefs["Coefficient Estimate"]=pd.Series(lreg.coef_)
coefs
#Standardizasyon
reg=Ridge(alpha=0.05)

scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train.values)
X_val_scaled=scaler.transform(X_val.values)
X_test_scaled=scaler.transform(X_test.values)

reg.fit(X_train_scaled,y_train)

pred = reg.predict(X_val_scaled)

#mean square error
mse = np.mean((pred - y_val)**2)

#r2

reg.score(X_val_scaled, y_val) #Ridge has better r-score and mse than linear reg.Lets check polynomial
mse
coefs=pd.DataFrame(X_train.columns)
coefs["Coefficient Estimate"]=pd.Series(reg.coef_)
coefs
poly=PolynomialFeatures(degree=2)
lm_poly=LinearRegression()

X_train_poly=poly.fit_transform(X_train.values)
X_val_poly=poly.transform(X_val.values)
X_test_poly=poly.transform(X_test.values)

lm_poly.fit(X_train_poly,y_train)

pred = lm_poly.predict(X_val_poly)

#mean square error
mse = np.mean((pred - y_val)**2)

#r2

lm_poly.score(X_val_poly, y_val)
mse
coefs=pd.DataFrame(X_train.columns)
coefs["Coefficient Estimate"]=pd.Series(lm_poly.coef_)
coefs
lass=Ridge(alpha=0.05)


lass.fit(X_train_scaled,y_train)

pred = lass.predict(X_val_scaled)

#mean square error
mse = np.mean((pred - y_val)**2)

#r2

lass.score(X_val_scaled, y_val)
coefs=pd.DataFrame(X_train.columns)
coefs["Coefficient Estimate"]=pd.Series(lass.coef_)
coefs
ridge_cv=RidgeCV(alphas=(0.0001,0.0005,0.001,0.01,0.005,0.05),normalize=True,cv=kf).fit(X_train,y_train)
ridge_cv_pred=ridge_cv.predict(X_val)

#mean square error
mse = np.mean((ridge_cv_pred - y_val)**2)

#r2

ridge_cv.score(X_val, y_val)
lasso_cv=LassoCV(alphas=(0.0001,0.0005,0.001,0.01,0.005,0.05),normalize=True,cv=kf).fit(X_train,y_train)
lasso_cv_pred=lasso_cv.predict(X_val)

#mean square error
mse = np.mean((lasso_cv_pred - y_val)**2)

#r2

lasso_cv.score(X_val, y_val)
print(f'Linear Regression val R^2: {lreg.score(X_val, y_val):.3f}')
print(f'Poly Regression val R^2: {lm_poly.score(X_val_poly, y_val):.3f}')
print(f'Ridge Regression val R^2: {reg.score(X_val_scaled, y_val):.3f}')
print(f'Lasso Regression val R^2: {lass.score(X_val_scaled, y_val):.3f}')
poly=PolynomialFeatures(degree=2)
lm_poly=LinearRegression()

X=poly.fit_transform(X.values)


lm_poly.fit(X,y)

pred = lm_poly.predict(X_test_poly)

#mean square error
mse = np.mean((pred - y_test)**2)

#r2

lm_poly.score(X_test_poly, y_test)
from sklearn.model_selection import cross_val_score,KFold
kf=KFold(n_splits=5,shuffle=True,random_state=100)
cross_val_score(lm_poly,X,y,cv=kf,scoring="r2") # train+val dataset