import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import os

os.listdir('../input')
df=pd.read_csv('../input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv',decimal=',').drop_duplicates()

df.head()
df.shape
df.columns
df.groupby(['date'])
plt.figure(figsize=(30,10))

df.groupby(['date']).mean()['% Silica Concentrate'].plot()

plt.show()
plt.figure(figsize=(30,10))

df.groupby(['date']).mean()['% Iron Concentrate'].plot()

plt.show()
df.groupby(['% Silica Concentrate']).mean()
df.groupby(['% Iron Concentrate']).mean()
# deleting date column
df=df.drop(['date'],axis=1)
from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.3)
print('Train size',train.shape)

print('Test size',test.shape)
import missingno

missingno.matrix(train,figsize=(20,5))
import missingno

missingno.matrix(test,figsize=(20,5))
plt.figure(figsize=(30,30))

sns.heatmap(train.corr(),annot=True,linewidths=0.3)

plt.show()
train.columns[-1]
train.info()
train.describe()
import statsmodels.api as sm
# Deleting date columns and % Iron Concentration

y=train['% Silica Concentrate']

X=train.drop(['% Silica Concentrate'],axis=1)
# Backward Elimination

cols=list(X.columns)

pmax=1

while len(cols)>0:

    p=[]

    C=X[cols]

    xc=sm.add_constant(C)

    model=sm.OLS(y,xc).fit()

    p=pd.Series(model.pvalues.values[1:],index=cols)

    pmax=max(p)

    feature_with_p_max=p.idxmax()

    if pmax>0.05:

        cols.remove(feature_with_p_max)

    else:

        break

        

selected_cols=cols

print(selected_cols)
import statsmodels.api as sm

xc=sm.add_constant(X[selected_cols])

xc=xc.drop([],axis=1)

model=sm.OLS(y,xc).fit()

model.summary()
residuals=model.resid

sns.distplot(residuals)
import scipy.stats as stats

stats.probplot(residuals,plot=plt)

plt.show()
import statsmodels.tsa.api as smt

acf=smt.graphics.plot_acf(residuals,lags=40)

acf.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

pd.DataFrame({'vif':vif},index=X.columns).T
from sklearn import metrics
y=train['% Silica Concentrate']

X=train.drop(['% Silica Concentrate'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                        test_size=0.30, random_state=42)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,y_train)
y_train_pred=lr.predict(X_train)

print ("intercept:",lr.intercept_)

print ("n_coefficients:         ",lr.coef_)
print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
y_test_pred=lr.predict(X_test)
print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))
from sklearn.linear_model import Lasso,LassoCV

lasso=Lasso(alpha=0.001,normalize=True)
lasso.fit(X_train,y_train)
y_train_pred=lasso.predict(X_train)

print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
y_test_pred=lasso.predict(X_test)

print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))
n_alphas = 50

alphas = np.linspace(0.1,4.5, n_alphas)

coefs=[]

lasso = Lasso()

for a in alphas:

    lasso.set_params(alpha=a)

    lasso.fit(X, y)

    coefs.append(lasso.coef_)

    

plt.plot(alphas, coefs)

plt.xlabel('alphas')

plt.ylabel('coefs')

plt.show()
n_alphas=50

alphas=np.linspace(0.1,1, n_alphas)



lasso_cv = LassoCV(alphas=alphas, cv=3, random_state=22)

lasso_cv.fit(X,y)
lasso_cv.alpha_
lasso = Lasso(alpha=lasso_cv.alpha_)

lasso.fit(X_train, y_train)

lasso.coef_

pd.DataFrame(lasso.coef_, X.columns, columns=['coefs'])
y_train_pred=lasso.predict(X_train)

print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
y_test_pred=lasso.predict(X_test)

print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))
from sklearn.linear_model import Ridge,RidgeCV

ridge=Ridge(alpha=0.05)
ridge.fit(X_train,y_train)
y_train_pred=ridge.predict(X_train)

print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
y_test_pred=ridge.predict(X_test)

print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))
ridge.fit(X, y)

ridge.coef_

pd.DataFrame(ridge.coef_, X.columns, columns=['coefs'])
n_alphas = 200

alphas = np.logspace(-3, 2, n_alphas)

coefs=[]

model = Ridge()

for a in alphas:

    model.set_params(alpha=a)

    model.fit(X, y)

    coefs.append(model.coef_)

    

plt.plot(alphas, coefs)

plt.xlabel('alphas')

plt.ylabel('coefs')

plt.show()
n_alphas = 1000

alphas = np.logspace(-2, 0)



ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)

ridge_cv.fit(X, y)



ridge_cv.alpha_
ridge=Ridge(alpha=ridge_cv.alpha_)

ridge.fit(X_train,y_train)
y_train_pred=ridge.predict(X_train)

print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
y_test_pred=ridge.predict(X_test)

print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))
from sklearn.linear_model import ElasticNet, ElasticNetCV

enet = ElasticNet(alpha=0.1)

enet.fit(X_train, y_train)
y_train_pred=enet.predict(X_train)

print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
y_test_pred=enet.predict(X_test)

print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))
pd.DataFrame(enet.coef_, X.columns, columns=['coefs'])
n_alphas = 100

alphas = np.logspace(-3, -1, n_alphas)

coefs=[]

enet = ElasticNet()

for a in alphas:

    enet.set_params(alpha=a)

    enet.fit(X, y)

    coefs.append(model.coef_)

    

plt.plot(alphas, coefs)

plt.xlabel('alphas')

plt.ylabel('coefs')

plt.show()
n_alphas = 100

alphas = np.logspace(-3, 1, n_alphas)



en_cv = ElasticNetCV(alphas=alphas, cv=3)

en_cv.fit(X, y)

en_cv.alpha_
enet = ElasticNet(alpha=en_cv.alpha_)

enet.fit(X_train,y_train)
y_train_pred=enet.predict(X_train)

print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
y_train_pred=enet.predict(X_train)

print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
from sklearn.tree import DecisionTreeRegressor

train_accuracy=[]

test_accuracy=[]



for depth in range(5,20):

    dt_model=DecisionTreeRegressor(max_depth=depth,random_state=42)

    dt_model.fit(X_train,y_train)

    train_accuracy.append(dt_model.score(X_train,y_train))

    test_accuracy.append(dt_model.score(X_test,y_test))
frame=pd.DataFrame({'max_depth':range(5,20),'train_accuracy':train_accuracy,'test_accuracy':test_accuracy})

print(frame)
plt.figure(figsize=(13,6))

plt.plot(frame['max_depth'],frame['train_accuracy'],marker='o')

plt.plot(frame['max_depth'],frame['test_accuracy'],marker='o')

plt.xlabel('Depth of Tree')

plt.ylabel('Accuracy Performance')

plt.show()
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
print ("feature_importances:",dtr.feature_importances_)

print ("Best params: \n        ",dtr.get_params)

print('n feature',dtr.n_features_)

print(dtr.n_outputs_)
y_train_pred=dtr.predict(X_train)

print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
y_test_pred=dtr.predict(X_test)

print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)
print ("feature_importances:",rfr.feature_importances_)

print ("n_coefficients:         ",rfr.get_params)

print('n feature',rfr.n_features_)

print(rfr.n_outputs_)
y_train_pred=rfr.predict(X_train)

print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
y_test_pred=rfr.predict(X_test)

print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))
test
expected_output=train['% Silica Concentrate']

expected_input=train.drop(['% Silica Concentrate'],axis=1)
y_pred=rfr.predict(expected_input)
print('R2 of Output: ', metrics.r2_score(expected_output,y_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(expected_output,y_pred))

print('Mean square Error: ',metrics.mean_squared_error(expected_output,y_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(expected_output,y_pred)))
params = {

    'n_estimators': 80,

    'max_depth': 12,

    'learning_rate': 0.1,

    'criterion': 'mse'

    }
from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor(**params)
gbr.fit(X_train,y_train)
y_train_pred=gbr.predict(X_train)

print('R2 of Train: ', metrics.r2_score(y_train,y_train_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_train,y_train_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_train,y_train_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_train,y_train_pred)))
y_test_pred=gbr.predict(X_test)

print('R2 of Test: ', metrics.r2_score(y_test,y_test_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(y_test,y_test_pred))

print('Mean square Error: ',metrics.mean_squared_error(y_test,y_test_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))
expected_output=train['% Silica Concentrate']

expected_input=train.drop(['% Silica Concentrate'],axis=1)
y_pred=gbr.predict(expected_input)
print('R2 of Output: ', metrics.r2_score(expected_output,y_pred))

print('Mean absolute Error: ',metrics.mean_absolute_error(expected_output,y_pred))

print('Mean square Error: ',metrics.mean_squared_error(expected_output,y_pred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(expected_output,y_pred)))