import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/kaggle/input/house-price-prediction/train.csv")
df.head()
df.isnull().sum()
df['LotFrontage']=df['LotFrontage'].replace(np.NaN,df['LotFrontage'].mean())

df.shape
df.info()
df.isnull().sum()
df=df.drop(["MiscFeature","Fence","PoolQC","Alley"],axis=1)
df.shape
df['FireplaceQu'].value_counts()
df['FireplaceQu']=df['FireplaceQu'].replace(np.NaN,'Gd')


df.isnull().values.sum()
df=df.dropna()
df.isnull().values.sum()
# Seperating the Categorical columns from the data
cat_data = df.select_dtypes(include=['object'])
p=cat_data.columns
import seaborn as sns
def fn(p):
    for i in p:
        sns.countplot(x=i, data=df)
        plt.title(i)
        plt.show()
fn(p)
def fn2(p):
    for i in p:
        s=df[i].value_counts()
        print(i)
        print(s)
fn2(p)
df=df.drop(['Street','Condition2','RoofStyle','Heating','Id'],axis=1)
df.shape

df.info()
int_data = df.select_dtypes(include=['int64'])
s=int_data.columns

def fn3(a):
    for i in a:
        sns.scatterplot(x=df[i],y=df['SalePrice'],data=df)
        
        plt.show()
fn3(s)
df = df[df['PoolArea']< 200]
df=df[df['BsmtHalfBath']<1.75]
df=df[df['BsmtFullBath']<2.5]
df=df[df['TotalBsmtSF']<4000]
df=df[df['1stFlrSF']<4000]
df.shape
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]

cat_data = X.select_dtypes(include=['object'])
p=cat_data.columns
dummies=pd.get_dummies(cat_data,drop_first=True)
dummies
X=X.drop(cat_data,axis=1)
X
X=pd.concat([X,dummies],axis=1)
X
X.isnull().values.sum()
from sklearn.preprocessing import StandardScaler

i_f_data = X.select_dtypes(include=['int64','float64'])
s=i_f_data.columns

scaler = StandardScaler()
X[s] = scaler.fit_transform(X[s])
X.head(5)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
regr=LinearRegression()
regr.fit(X_train,Y_train)


mse = np.mean((regr.predict(X_test) - Y_test)**2)
mse
print(metrics.r2_score(y_true=Y_train, y_pred=regr.predict(X_train)))
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regr=GridSearchCV(ridge,parameters)
ridge_regr.fit(X_train,Y_train)
print(ridge_regr.best_params_)
print(ridge_regr.best_score_)
print(metrics.r2_score(y_true=Y_train, y_pred=ridge_regr.predict(X_train)))
print(metrics.r2_score(y_true=Y_test, y_pred=ridge_regr.predict(X_test)))
sns.distplot(Y_test-ridge_regr.predict(X_test))
from sklearn.linear_model import Lasso
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso=Lasso()
lasso_regr=GridSearchCV(lasso,parameters)

lasso_regr.fit(X_train,Y_train)
print(lasso_regr.best_params_)
print(lasso_regr.best_score_)
print(metrics.r2_score(y_true=Y_train, y_pred=lasso_regr.predict(X_train)))
print(metrics.r2_score(y_true=Y_test, y_pred=lasso_regr.predict(X_test)))
sns.distplot(Y_test-lasso_regr.predict(X_test))