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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

from scipy import stats

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

from scipy import stats

import statsmodels.stats.api as sms

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import RFE

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV

from sklearn.metrics import r2_score
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')

df.head()
df.info()
df = pd.get_dummies(df, prefix = ["sex", "smoker", "region"], drop_first=True)
df.head()
fig, axes = plt.subplots(1,2, figsize=(14,5))



sns.distplot(df[df['smoker_yes']==1]['charges'], ax=axes[0], color='r').set_title('Distribution of Charges for Smokers')



sns.distplot(df[df['smoker_yes']==0]['charges'], ax=axes[1], color='g')

plt.title('Distribution of Charges for Non-Smokers')

plt.show()
plt.figure(figsize=(10,8))

sns.boxplot(df['sex_male'], df['charges'])

plt.grid()

plt.show()
plt.figure(figsize=(12,8))

sns.scatterplot(x='bmi', y='charges', data=df)

plt.show()
plt.figure(figsize=(15,7))

sns.violinplot(x='children', y='charges', data=df)

plt.show()
plt.figure(figsize=(10,5))

sns.barplot(df.groupby('children').mean()['charges'].index, df.groupby('children').mean()['charges'].values)

plt.grid()

plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(), annot=True)

plt.show()
X = df.drop('charges', axis=1)

y = df['charges']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)



ss = StandardScaler()



X_trains = ss.fit_transform(X_train)

X_tests = ss.transform(X_test)
Xc = sm.add_constant(X)

model = sm.OLS(y, Xc)

lr = model.fit()

lr.summary()
vif = [vif(Xc.values, i) for i in range(Xc.shape[1])]

pd.DataFrame(vif, index=Xc.columns, columns=['VIF'])
pred = lr.predict()

sns.regplot(x=pred, y=y, line_kws={'color':'red'})

plt.show()
fstat, pvalue = sm.stats.diagnostic.linear_rainbow(lr)

print("The p-value is: ",pvalue)
stats.probplot(lr.resid, plot=plt)

plt.show()
sns.distplot(lr.resid)

plt.show()
stat, pvalue = stats.jarque_bera(lr.resid)



print("The p-value is: ",pvalue)
sns.residplot(lr.predict(), lr.resid, lowess=True, line_kws={'color':'red'})

plt.show()
fval, pval, res = sms.het_goldfeldquandt(lr.resid, Xc)



print("The p-value is: ",pval)
while (len(X.columns)>0):

    Xc1 = sm.add_constant(X)

    ols = sm.OLS(y, Xc1)

    model = ols.fit()

    f = model.pvalues[1:].idxmax()

    if (model.pvalues[1:].max()>0.05):

        X = X.drop(f, axis=1)

    else:

        break



print("The final features are:",X.columns)
Xc2 = sm.add_constant(X)

ols = sm.OLS(y, Xc2)

lr = ols.fit()

lr.summary()
error = lr.resid

mse = np.mean(error**2)

rmse = np.sqrt(mse)

rmse
X = df.drop(['charges'], axis = 1)

y = df.charges



X_train,X_test,y_train,y_test = train_test_split(X,y, random_state = 0)

lr = LinearRegression().fit(X_train,y_train)



y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



print("The score is:",lr.score(X_test,y_test))

print("The RMSE for the training set is:",np.sqrt(mean_squared_error(y_train, y_train_pred)))

print("The RMSE for the testing set is:",np.sqrt(mean_squared_error(y_test, y_test_pred)))
quad = PolynomialFeatures (degree = 2)

x_quad = quad.fit_transform(X)



X_train,X_test,y_train,y_test = train_test_split(x_quad,y, random_state = 0)



plr = LinearRegression().fit(X_train,y_train)



y_train_pred = plr.predict(X_train)

y_test_pred = plr.predict(X_test)



rmseLinear = np.sqrt(mean_squared_error(y_test, y_test_pred))



print("The score is:",plr.score(X_test,y_test))

print("The RMSE for the training set is:",np.sqrt(mean_squared_error(y_train, y_train_pred)))

print("The RMSE for the testing set is:",np.sqrt(mean_squared_error(y_test, y_test_pred)))
lr = LinearRegression()

rfe = RFE(lr, n_features_to_select=4)

rfe.fit(X, y)

pd.DataFrame(rfe.ranking_, index=X.columns, columns=['Select'])
lr = LinearRegression()

param_grid = [{'n_features_to_select':list(range(1,len(df.columns)+1))}]



rfe = RFE(lr)

gsearch = GridSearchCV(rfe, param_grid=param_grid, cv=3, return_train_score=True)

gsearch.fit(X, y)
print(gsearch.best_params_)

pd.DataFrame(gsearch.cv_results_)
lr = LinearRegression()

rfe = RFE(lr, n_features_to_select=8)

rfe.fit(X, y)

pd.DataFrame(rfe.ranking_, index=X.columns, columns=['Rank'])
X_train, X_test, y_train, y_test = train_test_split(x_quad, y, random_state = 0)



lassoModel = Lasso(max_iter=5000)

lasso = lassoModel.fit(X_train, y_train)

lassoPred = lasso.predict(X_test)

mseLasso = mean_squared_error(y_test, lassoPred)

rmseLasso = mseLasso**(1/2)



print("The RMSE for the model is:",rmseLasso)

print("The Rsquare for the model is:",lasso.score(X_test, y_test))
ridgeModel = Ridge(max_iter=5000)

ridge = ridgeModel.fit(X_train, y_train)

ridgePred = ridge.predict(X_test)

mseRidge = mean_squared_error(y_test, ridgePred)

rmseRidge = mseRidge**(1/2)



print("The RMSE for the model is:",rmseRidge)

print("The Rsquare for the model is:",ridge.score(X_test, y_test))
elasticNetModel = ElasticNet(alpha = 0.01, l1_ratio = 0.9, max_iter = 5000)

ElasticNet = elasticNetModel.fit(X_train, y_train)

ElasticNetPred = ElasticNet.predict(X_test)

mseElasticNet = mean_squared_error(y_test, ElasticNetPred)

rmseElasticNet = mseElasticNet**(1/2)



print("The RMSE for the model is:",rmseElasticNet)

print("The Rsquare for the model is:",ElasticNet.score(X_test, y_test))
performanceData = pd.DataFrame({"Regrssion":["Linear", "Lasso", "Ridge", "Elasticnet"], 

                                "RMSE":[rmseLinear, rmseLasso, rmseRidge, rmseElasticNet]})

performanceData