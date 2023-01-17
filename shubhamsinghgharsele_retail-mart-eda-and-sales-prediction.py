



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import statsmodels.api as sm



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.linear_model import ElasticNet, Lasso, RidgeCV

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error



import xgboost as xgb

import lightgbm as lgb



from math import sqrt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("../input/Train.csv")

data_test = pd.read_csv("../input/Test.csv")
data_train.head()
data_test.head()
print("Training Data  Row : %s Column : %s " % (str(data_train.shape[0]) ,str(data_train.shape[1])))
print("Training Data  Row : %s Column : %s " % (str(data_test.shape[0]) ,str(data_test.shape[1])))
data_train.info()
data_test.info()
data_train.isnull().sum()
data_test.isnull().sum()
data_train['Item_Weight'] = data_train['Item_Weight'].fillna((data_train['Item_Weight'].mean()))

data_train.Outlet_Size = data_train.Outlet_Size.fillna("Small")

data_train.isnull().sum().sum()
data_test['Item_Weight'] = data_test['Item_Weight'].fillna((data_test['Item_Weight'].mean()))

data_test.Outlet_Size = data_test.Outlet_Size.fillna("Small")

data_test.isnull().sum().sum()
column_num = data_train.select_dtypes(exclude = ["object"]).columns

column_object = data_train.select_dtypes(include = ["object"]).columns



test_column_num = data_test.select_dtypes(exclude = ["object"]).columns

test_column_object = data_test.select_dtypes(include = ["object"]).columns
data_train_num = data_train[column_num]

data_train_object = data_train[column_object]



data_test_num = data_test[test_column_num]

data_test_object = data_test[test_column_object]
data_train_num.describe()
data_train_object.describe()
sns.distplot(data_train_num["Item_Outlet_Sales"]);
print("Skewness: %f" % data_train_num["Item_Outlet_Sales"].skew())

print("Kurtosis: %f" % data_train_num["Item_Outlet_Sales"].kurt())
%matplotlib inline

data_train_num.hist(figsize=(10,8),bins=6,color='Y')

plt.tight_layout()

plt.show()
plt.figure(1)

plt.subplot(321)

data_train['Outlet_Type'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='green')



plt.subplot(322)

data_train['Item_Fat_Content'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='yellow')



plt.subplot(323)

data_train['Item_Type'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='red')



plt.subplot(324)

data_train['Outlet_Size'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='orange')



plt.subplot(325)

data_train['Outlet_Location_Type'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='black')



plt.subplot(326)

data_train['Outlet_Establishment_Year'].value_counts().plot(figsize=(10,12),kind='bar',color='olive')





plt.tight_layout()

plt.show()
data_train['Item_Fat_Content'].value_counts()
data_test['Item_Fat_Content'].value_counts()
vals_to_replace = {'LF':'Low Fat', 'low fat':'Low Fat', 'reg':'Regular'}

data_train['Item_Fat_Content'] = data_train['Item_Fat_Content'].map(vals_to_replace)

data_test['Item_Fat_Content'] = data_test['Item_Fat_Content'].map(vals_to_replace)
data_train['Item_Fat_Content'].value_counts(normalize=True).plot(figsize=(5,4),kind='bar',color='green')
data_train["Outlet_Identifier"].value_counts()


ax = sns.catplot(x="Outlet_Identifier", y = "Item_Outlet_Sales", data=data_train, height=5, aspect=2,  kind="bar")

plt.rcParams['figure.figsize']=(10,4)

ax = sns.boxplot(x="Outlet_Type", y="Item_Outlet_Sales", data=data_train)

ax = sns.boxplot(x="Outlet_Size", y="Item_Outlet_Sales", data=data_train)
ax = sns.boxplot(x="Item_Fat_Content", y="Item_Outlet_Sales", data=data_train)
sns.pairplot(data_train[data_train_num.columns])
sns.heatmap(data_train[data_train_num.columns].corr(),annot=True)



total_object = data_train_object.append(data_test_object)

train_object_lenght = len(data_train_object)

total_cat = pd.get_dummies(total_object, drop_first= True)

data_train_object = total_cat[:train_object_lenght]

data_test_object = total_cat[train_object_lenght:]
data_train_object.head()
df_test = pd.concat([data_test_object,data_test_num],axis=1)

df_train = pd.concat([data_train_object,data_train_num],axis=1)

df_train.head()
train_Y = df_train.iloc[:,-1]

train_X=  df_train.iloc[:,0:-1]
model = sm.OLS(train_Y, train_X)

results = model.fit()

print(results.summary())


scaler = StandardScaler()

train_scaler = scaler.fit(train_X)

train_scale = train_scaler.transform(train_X)

train_X = pd.DataFrame(train_scale, columns=train_X.columns)



train_scale = train_scaler.transform(df_test)

df_test = pd.DataFrame(train_scale, columns=df_test.columns)
model = sm.OLS(train_Y, train_X)

results = model.fit()

print(results.summary())
type(results.pvalues.index)
type(results.pvalues)
col = [value for value in results.pvalues.index if results.pvalues[value] > -0.001  ]

col
X1_train, X1_test, Y1_train,Y1_test =  train_test_split(train_X[col],train_Y, random_state=33)

print(X1_train.shape)

print(X1_test.shape)

print(df_test.shape)
pca_model = PCA(n_components=0.95)

X1_train = pca_model.fit_transform(X1_train)

X1_test = pca_model.transform(X1_test)

test_X = pca_model.transform(df_test[col])

print(X1_train.shape)

print(X1_test.shape)

print(test_X.shape)
def modelPredection(model,X1_train,Y1_train,X1_test,Y1_test,test_X) :

    model.fit(X1_train,Y1_train)

    Y1_predict = model.predict(X1_test)

    print("RMSE : %f"%sqrt(mean_squared_error(Y1_test,Y1_predict)))

    return model.predict(test_X)
#Linear Regression

linear = linear_model.LinearRegression( fit_intercept=True, n_jobs=None,

         normalize=False);

predict_Y = modelPredection(linear,X1_train,Y1_train,X1_test,Y1_test,test_X)
#RidgeCV

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X1_train,Y1_train)



predict_Y = modelPredection(clf,X1_train,Y1_train,X1_test,Y1_test,test_X)
#Kernel Ridge

RR = KernelRidge(alpha=0.6, kernel='polynomial', degree=3, coef0=2.5)

predict_Y = modelPredection(RR,X1_train,Y1_train,X1_test,Y1_test,test_X)
#Lasso

#lasso = Lasso(alpha =1.1, random_state=1)

lasso = Lasso(alpha =16, random_state=100)

predict_Y = modelPredection(lasso,X1_train,Y1_train,X1_test,Y1_test,test_X)
#Elastic Net 

#elastic_net = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)

elastic_net = ElasticNet(alpha=0.8)

predict_Y = modelPredection(elastic_net,X1_train,Y1_train,X1_test,Y1_test,test_X)
#Gradient Boosting

#GBR = GradientBoostingRegressor(n_estimators=30, max_depth=2)

GBR = GradientBoostingRegressor()

predict_Y = modelPredection(GBR,X1_train,Y1_train,X1_test,Y1_test,test_X)
#XGB

model_xgb = xgb.XGBRegressor()

predict_Y = modelPredection(model_xgb,X1_train,Y1_train,X1_test,Y1_test,test_X)
#light Gradient Boosting

model_lgb = lgb.LGBMRegressor()

predict_Y = modelPredection(model_lgb,X1_train,Y1_train,X1_test,Y1_test,test_X)