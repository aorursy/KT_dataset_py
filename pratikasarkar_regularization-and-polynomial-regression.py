import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

df = load_boston()

X = pd.DataFrame(df.data,columns=df.feature_names)
X
y = df.target
for i in X.columns:

    sns.scatterplot(i,y,data = X)

    plt.show()
plt.figure(figsize = (10,7))

sns.heatmap(X.corr(),annot = True)
import statsmodels.api as sm

Xc = sm.add_constant(X)

linreg = sm.OLS(y,Xc).fit()

linreg.summary()
Xc.drop(['INDUS','AGE'],axis = 1,inplace = True)
linreg = sm.OLS(y,Xc).fit()

linreg.summary()
linreg.rsquared
np.sqrt(linreg.mse_resid)
X = Xc.drop('const',axis = 1)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X,y)
model.score(X,y)
y = pd.DataFrame(y,columns=['medv'])
from sklearn.model_selection import KFold

from sklearn import metrics

kf = KFold(n_splits=5,shuffle=True,random_state=0)

for model,name in zip([model],['Linear_Regression']):

    rmse = []

    for train_idx,test_idx in kf.split(X,y):

        X_train,X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]

        y_train,y_test = y.iloc[train_idx,:],y.iloc[test_idx,:]

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        mse = metrics.mean_squared_error(y_test,y_pred)

        print(np.sqrt(mse))

        rmse.append(np.sqrt(mse))

    print('RMSE scores : %0.03f (+/- %0.05f) [%s]'%(np.mean(rmse), np.var(rmse,ddof = 1), name))
from sklearn.linear_model import Ridge,Lasso,ElasticNet

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
m1 = LinearRegression()

m2 = Ridge(alpha=0.5,normalize=True) # Scaling is mandatory for all distance based calculations

m3 = Lasso(alpha=0.1,normalize=True)

m4 = ElasticNet(alpha=0.01,l1_ratio=0.92,normalize=True)
# from sklearn.model_selection import GridSearchCV

# params = { 'alpha' : np.arange(0.01,1,0.01) }

# #           ,'l1_ratio' : np.arange(0.1,1,0.01)}

# gscv = GridSearchCV(m3,params,cv = 5,scoring = 'neg_mean_squared_error')

# gscv.fit(X,y)

# gscv.best_params_
model = m1.fit(X_train,y_train)

sns.barplot(x = X.columns,y = sorted(model.coef_[0]))

plt.title('LR coefficients')
model = m2.fit(X_train,y_train)

sns.barplot(x = X.columns,y = sorted(model.coef_[0]))

plt.title('Ridge coefficients')
model = m3.fit(X_train,y_train)

sns.barplot(x = X.columns,y = sorted(model.coef_))

plt.title('LASSO coefficients')
model = m4.fit(X_train,y_train)

sns.barplot(x = X.columns,y = sorted(model.coef_))

plt.title('ElasticNet coefficients')
from sklearn.model_selection import KFold

from sklearn import metrics

kf = KFold(n_splits=5,shuffle=True,random_state=0)

for model,name in zip([m1,m2,m3,m4],['Linear_Regression','Ridge','LASSO','ElasticNet']):

    rmse = []

    for train_idx,test_idx in kf.split(X,y):

        X_train,X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]

        y_train,y_test = y.iloc[train_idx,:],y.iloc[test_idx,:]

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        mse = metrics.mean_squared_error(y_test,y_pred)

        rmse.append(np.sqrt(mse))

    print('RMSE scores : %0.03f (+/- %0.05f) [%s]'%(np.mean(rmse), np.var(rmse,ddof = 1), name))

    print()
print('Bias error increased after Lasso : ',(5.875-4.829)/5.875 * 100,"%")
print('Variance error decreased after Lasso : ',(0.53924 - 0.41470)/0.53924 * 100,"%")
mpg_df = pd.read_csv('../input/auto-mpg-pratik.csv')
mpg_df.head()
sns.pairplot(mpg_df)
X_update = mpg_df.drop(['mpg', 'cylinders', 'displacement', 'horsepower',

       'acceleration', 'car name'],axis = 1)

y = mpg_df['mpg']
from sklearn.preprocessing import PolynomialFeatures

qr = PolynomialFeatures(degree=2)

x_qr = qr.fit_transform(X_update[['weight']])

x_qr = x_qr[:,2:]

x_qr_df = pd.DataFrame(x_qr,columns=['weight_square'])
df_final = pd.concat([X_update,x_qr_df,y],axis = 1)

df_final
import statsmodels.api as sm

X = df_final.drop('mpg',axis = 1)

y = df_final['mpg']

Xc = sm.add_constant(X)

lr = sm.OLS(y,Xc).fit()

lr.summary()
from sklearn.linear_model import LinearRegression

model = LinearRegression()

m2 = Ridge(alpha=0.06,normalize=True) # Scaling is mandatory for all distance based calculations

m3 = Lasso(alpha=0.37,normalize=True)

m4 = ElasticNet(alpha=0.01,l1_ratio=0.1,normalize=True)
y = pd.DataFrame(y,columns=['mpg'])
from sklearn.model_selection import KFold

from sklearn import metrics

kf = KFold(n_splits=5,shuffle=True,random_state=0)

for model,name in zip([model,m2,m3,m4],['Quadratic_Regression','Ridge','Lasso','ElasticNet']):

    rmse = []

    for train_idx,test_idx in kf.split(X,y):

        X_train,X_test = X.iloc[train_idx,:],X.iloc[test_idx,:]

        y_train,y_test = y.iloc[train_idx,:],y.iloc[test_idx,:]

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        mse = metrics.mean_squared_error(y_test,y_pred)

        rmse.append(np.sqrt(mse))

    print('RMSE scores : %0.03f (+/- %0.05f) [%s]'%(np.mean(rmse), np.std(rmse,ddof = 1), name))

    print()
print('Bias error increased after Ridge : ',(3.409-3.021)/3.409 * 100,"%")
print('Variance error decreased after Ridge : ',(0.36898 - 0.31105)/0.36898 * 100,"%")