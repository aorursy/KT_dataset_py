import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew
df = pd.read_csv('../input/abalone-dataset/abalone.csv')

df.head()
df.info()
df.describe()
df.shape
df['Age'] = df['Rings']+1.5

df = df.drop('Rings', axis=1)

df.head()
df.describe()
df.hist(figsize=(20,10), grid=False, layout=(2,4), bins=30)

plt.show()
Numerical = df.select_dtypes(include=[np.number]).columns

Categorical = df.select_dtypes(include=[np.object]).columns
skew_values = skew(df[Numerical], nan_policy = 'omit')

dummy = pd.concat([pd.DataFrame(list(Numerical), columns=['Features']), 

           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1)

dummy.sort_values(by = 'Skewness degree' , ascending = False)
plt.figure(figsize=(10,5))

sns.countplot(df['Sex'])

plt.show()
df.boxplot(figsize=(20,10))

plt.show()
plt.figure(figsize=(10,5))

sns.distplot(df['Age'])

plt.show()
df.head()
fig, axes = plt.subplots(4,2, figsize=(15,15))

axes = axes.flatten()



for i in range(1,len(df.columns)-1):

    sns.scatterplot(x=df.iloc[:,i], y=df['Age'], ax=axes[i])



plt.show()
plt.figure(figsize=(10,5))

sns.boxenplot(y=df['Age'], x=df['Sex'])

plt.grid()

plt.show()



df.groupby('Sex')['Age'].describe()
plt.figure(figsize=(10,5))

sns.heatmap(df.corr(), annot=True)

plt.show()
sns.pairplot(df)

plt.show()
df = pd.get_dummies(df, drop_first=True)

df.head()
X = df.drop(['Age'], axis=1)

y = df['Age']



import statsmodels.api as sm



Xc = sm.add_constant(X)

lr = sm.OLS(y, Xc).fit()

lr.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF



vif = [VIF(Xc.values, i) for i in range(Xc.shape[1])]

pd.DataFrame(vif, index=Xc.columns, columns=['VIF'])
X2 = X.drop(['Whole weight'], axis=1)

X2c = sm.add_constant(X2)



vif = [VIF(X2c.values, i) for i in range(X2c.shape[1])]

pd.DataFrame(vif, index=X2c.columns, columns=['VIF'])
X2 = X.drop(['Whole weight','Diameter'], axis=1)

X2c = sm.add_constant(X2)



vif = [VIF(X2c.values, i) for i in range(X2c.shape[1])]

pd.DataFrame(vif, index=X2c.columns, columns=['VIF'])
X2 = X.drop(['Whole weight','Diameter','Viscera weight'], axis=1)

X2c = sm.add_constant(X2)



vif = [VIF(X2c.values, i) for i in range(X2c.shape[1])]

pd.DataFrame(vif, index=X2c.columns, columns=['VIF'])
lr = sm.OLS(y, X2c).fit()

lr.summary()
y_pred = lr.predict(X2c)



plt.figure(figsize=(10,5))

sns.regplot(y_pred, y, lowess=True, line_kws={'color':'red'})

plt.show()
stat, pval = sm.stats.diagnostic.linear_rainbow(res=lr, frac=0.5)

print(pval)
# QQ Plot



from scipy import stats



resid = lr.resid



plt.figure(figsize=(10,5))

stats.probplot(resid, plot=plt)

plt.show()
from scipy.stats import norm

norm.fit(resid)



plt.figure(figsize=(10,5))

sns.distplot(resid, fit=norm)

plt.show()
stat, pval = stats.jarque_bera(resid)

print(pval)
fig, axes = plt.subplots(2,2, figsize=(15,18))

axes = axes.flatten()



for i in range(len(X2.columns)-2):

    sns.distplot(X2.iloc[:,i], ax=axes[i])



plt.show()
plt.figure(figsize=(10,5))

sns.residplot(lr.predict(),lr.resid)

plt.show()
import statsmodels.stats.api as sms

sms.het_goldfeldquandt(lr.resid, X2c)
while len(X2.columns)>0:

    X_c = sm.add_constant(X2)

    mod = sm.OLS(y,X_c).fit()

    f = mod.pvalues[1:].idxmax()

    if mod.pvalues[1:].max()>0.05:

        X2 = X2.drop(f, axis=1)

    else:

        break



print("The final features are:",X2.columns)
mod.summary()
err = mod.resid

mse = np.mean(err**2)

rmse = np.sqrt(mse)



print("The root mean Sq error derived fro the statistical summary is:",rmse)
df.head()
X = df.drop('Age', axis=1)

y = df['Age']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



from sklearn.preprocessing import StandardScaler

ss = StandardScaler()



X_trains = ss.fit_transform(X_train)

X_tests = ss.transform(X_test)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()



lr.fit(X_trains, y_train)

pred = lr.predict(X_tests)



from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, pred))

r2 = r2_score(y_test, pred)



print("The root mean Sq error calculated from the base model is:",rmse)

print("The r2-score is:",r2)
from sklearn.feature_selection import RFE

lr = LinearRegression()

n = [{'n_features_to_select':list(range(1,10))}]

rfe = RFE(lr)



from sklearn.model_selection import GridSearchCV

gsearch = GridSearchCV(rfe, param_grid=n, cv=3)

gsearch.fit(X, y)



gsearch.best_params_
lr = LinearRegression()

rfe = RFE(lr, n_features_to_select=8)

rfe.fit(X,y)



pd.DataFrame(rfe.ranking_, index=X.columns, columns=['Rank'])
from sklearn.linear_model import Lasso, LassoCV



lasso = Lasso(alpha=0.1)

lasso.fit(X,y)

pd.DataFrame(lasso.coef_, index=X.columns, columns=['Coefs'])
alphas = np.linspace(0.001, 0.1, 100)

lassocv = LassoCV(alphas=alphas, cv=3, random_state=1, max_iter=5000)

lassocv.fit(X,y)

lassocv.alpha_
lasso = Lasso(alpha=lassocv.alpha_, max_iter=5000)

lasso.fit(X,y)

pd.DataFrame(lasso.coef_, index=X.columns, columns=['Coefs'])
from sklearn.model_selection import cross_val_score



res = cross_val_score(lasso, X, y, cv=3, scoring='neg_mean_squared_error')

rmse_lasso = np.sqrt(abs(res))

print("The RMSE for Lasso regression is:",rmse_lasso.mean())
from sklearn.linear_model import Ridge, RidgeCV

ridge = Ridge(alpha=0.5)

ridge.fit(X, y)

pd.DataFrame(ridge.coef_, index=X.columns, columns=['Coefs'])
alphas = np.logspace(-3,1,1000)

coefs = []

for a in alphas:

    model = Ridge(alpha=a)

    model.fit(X,y)

    coefs.append(model.coef_)



plt.figure(figsize=(10,5))    

plt.plot(alphas, coefs)

plt.show()
alphas = np.logspace(-2,0,1000)

ridgecv = RidgeCV(alphas=alphas, cv=3)

ridgecv.fit(X,y)

ridgecv.alpha_
ridge = Ridge(alpha=ridgecv.alpha_)

ridge.fit(X,y)

pd.DataFrame(ridge.coef_, index=X.columns, columns=['Coefs'])
res = cross_val_score(ridge, X, y, cv=3, scoring='neg_mean_squared_error')

rmse_ridge = np.sqrt(abs(res))

print("The RMSE for Ridge regression is:",rmse_ridge.mean())
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import  RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import  GradientBoostingRegressor

from sklearn.linear_model import  Ridge

from sklearn.svm import SVR

from sklearn import model_selection

from sklearn.model_selection import cross_val_predict



models = [   SVR(),

             RandomForestRegressor(),

             GradientBoostingRegressor(),

             KNeighborsRegressor(n_neighbors = 4)]

results = []

names = ['SVM','Random Forest','Gradient Boost','K-Nearest Neighbors']

for model,name in zip(models,names):

    kfold = model_selection.KFold(n_splits=10)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)

    rmse = np.sqrt(mean_squared_error(y, cross_val_predict(model, X , y, cv=3)))

    results.append(rmse)

    names.append(name)

    msg = "%s: %f" % (name, rmse)

    print(msg)