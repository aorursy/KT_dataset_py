import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as smf
Data=pd.read_csv('../input/housing (1).csv',index_col=0)
Data.head()
corr=Data.corr()
plt.figure(figsize=(10,5))

sns.heatmap(corr,annot=True)
import statsmodels.formula.api as smf
Data.columns
m1=smf.ols('medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat',Data).fit()
m1.summary()
m2=smf.ols('medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+black+lstat',Data).fit()
m2.summary()
#sns.pairplot(Data)
sns.pairplot(Data[['indus','age','medv']])
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
x=Data.drop(['medv','age','indus'],axis=1)

y=Data[['medv']]
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=1)
x_train=X_train[['lstat','rm']]
qr=PolynomialFeatures(degree=2,include_bias=False)
x_qr=qr.fit_transform(x_train)

x_qr
x_qr_df=pd.DataFrame(x_qr)
x_qr_df.shape
Y_train.shape
idx=np.arange(len(Y_train))
Y_train.index=idx
x_qr_df=pd.concat([x_qr_df,Y_train],axis=1)
x_qr_df.shape
x_qr_df.columns=['lstat','rm','lstat2','lstatXrm','rm2','medv']
x_qr_df.head()
m2=smf.ols('medv~rm+lstat2+lstatXrm+rm2',x_qr_df).fit()
m2.summary()
x_test=X_test[['lstat','rm']]
xtest_qr=qr.fit_transform(x_test)

xtest_qr_df=pd.DataFrame(xtest_qr)
xtest_qr_df.columns=['lstat','rm','lstat2','lstatXrm','rm2']
xtest_qr_df.head()
QR_pred=m2.predict(xtest_qr_df)
plt.plot(QR_pred,Y_test,'*')
from sklearn import metrics
MSE=metrics.mean_squared_error(QR_pred,Y_test)
QR_RMSE=np.sqrt(np.mean(MSE))

QR_RMSE
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(x_train,Y_train)
li_pred=model.predict(x_test)
plt.plot(li_pred,Y_test,'*')
MSE=metrics.mean_squared_error(li_pred,Y_test)
li_RMSE=np.sqrt(np.mean(MSE))

li_RMSE
R2=model.score(x_test,Y_test)

R2
pr=PolynomialFeatures(degree=3,include_bias=False)
x_pr=pr.fit_transform(x_train)

x_pr
x_pr_df=pd.DataFrame(x_pr)
x_pr_df.head()
x_pr_df.columns=['lstat','rm','lstat2','lstatXrm','rm2','lstat3','lstat2Xrm','lstatXrm2','rm3']
x_pr_df.head()
x_pr_df=pd.concat([x_pr_df,Y_train],axis=1)
x_pr_df.head()
x_pr_df.columns
mp1=smf.ols('medv~lstat+rm+lstatXrm+rm2+lstatXrm2+rm3',x_pr_df).fit() # from summarey we drop lstat2 lastat3 lsatat2rm

mp1.summary()
xtest_pr=pr.fit_transform(x_test)
xtest_pr_df=pd.DataFrame(xtest_pr)
xtest_pr_df.columns=['lstat','rm','lstat2','lstatXrm','rm2','lstat3','lstat2Xrm','lstatXrm2','rm3']
xtest_pr_df.head()
PR_pred=mp1.predict(xtest_pr_df)
plt.plot(PR_pred,Y_test,'*')
MSE=metrics.mean_squared_error(PR_pred,Y_test)
PR_RMSE=np.sqrt(np.mean(MSE))

PR_RMSE
from sklearn.linear_model import Ridge,Lasso
rd=Ridge(alpha=0.5,normalize=True)

rd.fit(X_train,Y_train)
rd_pred=rd.predict(X_test)
ls=Lasso(alpha=0.05,normalize=True)

ls.fit(X_train,Y_train)
ls_pred=ls.predict(X_test)
rd.coef_
ls.coef_
Variable=X_test.columns

Variable
ridge=pd.Series(rd.coef_,Variable).sort_values()
ridge.plot(kind='bar')
lasso=pd.Series(ls.coef_,Variable).sort_values()
lasso.plot(kind='bar')
from sklearn import metrics

from sklearn.model_selection import KFold

LR=LinearRegression(normalize=True)

ridge_R=Ridge(alpha=0.5,normalize=True)

lasso_L=Lasso(alpha=0.1,normalize=True)
kf=KFold (n_splits=3, shuffle=True, random_state=2)

for model, name in zip([LR,ridge_R,lasso_L],['MVLR','RIDGE','LASSO']):

    rmse=[]

    for train,test in kf.split(x,y):

        X_train,X_test=x.iloc[train,:],x.iloc[test,:]

        Y_train,Y_test=y[train],y[test]

        model.fit(X_train,Y_train)

        Y_pred=model.predict(X_test)

        rmse.append(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))

    print(rmse)

    print("cross VALIDATE RMSE score %0.03f (+/-%0.05f)[%s]"% (np.mean(rmse),np.var(rmse,ddof=1),name))