import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





import statsmodels.api as sm

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression,SGDRegressor

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import PolynomialFeatures
df=pd.read_csv('../input/medical-cost/insurance.csv')

df.head()
print(df.isnull().sum()[df.columns[df.isnull().sum()>0]])

print(df.shape)
df.describe().T
plt.figure(figsize=(15,5))

plt.subplot(121)

sns.countplot(df['sex'])



plt.subplot(122)

sns.countplot(df['smoker'],hue=df['sex'])

plt.show()



plt.figure(figsize=(15,5))

sns.countplot(df['region'])

plt.show()
plt.figure(figsize=(15,10))

sns.boxplot(df['smoker'],df['age'],hue=df['region'])
plt.figure(figsize=(10,5))

sns.barplot(df['region'],df['charges'])
plt.figure(figsize=(10,5))

sns.barplot(df['children'],df['charges'],ci=None)
plt.figure(figsize=(18,5))

plt.subplot(131)

sns.boxplot(df['sex'],df['charges'])



plt.subplot(1,3,3)

sns.boxplot(df['sex'],df['charges'],hue=df['smoker'])

plt.show()



plt.figure(figsize=(15,7))

sns.boxplot(df['sex'],df['bmi'],hue=df['smoker'])

plt.show()
sns.distplot(df['age'],bins=10)
sns.pairplot(df,hue='smoker')
plt.figure(figsize=(20,7))

plt.subplot(121)

sns.scatterplot(y=df['charges'],x=df['age'])



plt.subplot(122)

plt.hist(df['charges'],bins=11)

plt.show()
df.head()
cat_cols=df.select_dtypes(include=object).columns

for col in cat_cols:

    a=pd.get_dummies(df[col],drop_first=True)

    df=pd.concat((df,a),1)

    df=df.drop(col,1)
df=df.transform(lambda x: np.log1p(x))
df.head()
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True)
X=df.drop('charges',1)

y=df['charges']
lr=LinearRegression()

rfe=RFE(lr,4)
rfe=rfe.fit(X,y)

X_rfe4=X[X.columns[rfe.support_]]

y=y



X_train_rfe4,X_test_rfe4,y_train_rfe4,y_test_rfe4=train_test_split(X_rfe4,y,test_size=0.3,random_state=45)



#selected columns

print(X_train_rfe4.columns.values)
model_linear4=lr.fit(X_train_rfe4,y_train_rfe4)



print(f'The R square for train data is {round(model_linear4.score(X_train_rfe4,y_train_rfe4)*100,2)}')

print(f'The R square for test data is {round(model_linear4.score(X_test_rfe4,y_test_rfe4)*100,2)}')

print(f'The MSE for train data is {mean_squared_error(y_train_rfe4,model_linear4.predict(X_train_rfe4))}')

print(f'The MSE for test data is {mean_squared_error(y_test_rfe4,model_linear4.predict(X_test_rfe4))}')
#Checking the Adjusted R square value

X_train_rfe4_const=sm.add_constant(X_train_rfe4)

model_linear4_ols=sm.OLS(y_train_rfe4,X_train_rfe4_const).fit()

print(round((model_linear4_ols.rsquared)*100,2))
#Lets try with RFE of 5
rfe=RFE(lr,5)

rfe=rfe.fit(X,y)

X_rfe5=X[X.columns[rfe.support_]]

y=y



X_train_rfe5,X_test_rfe5,y_train_rfe5,y_test_rfe5=train_test_split(X_rfe5,y,test_size=0.3,random_state=45)



#selected columns

print(X_train_rfe5.columns.values)
model_linear5=lr.fit(X_train_rfe5,y_train_rfe5)



print(f'The R square for train data is {round(model_linear5.score(X_train_rfe5,y_train_rfe5)*100,2)}')

print(f'The R square for test data is {round(model_linear5.score(X_test_rfe5,y_test_rfe5)*100,2)}')

print(f'The MSE for train data is {mean_squared_error(y_train_rfe5,model_linear5.predict(X_train_rfe5))}')

print(f'The MSE for test data is {mean_squared_error(y_test_rfe5,model_linear5.predict(X_test_rfe5))}')
#Looks Like model with 5 features makes a good prediction as adjusted r square has increased. So lets stick on to 5 features
X_train_const=sm.add_constant(X_train_rfe5)

vif=[VIF(X_train_const.values,i) for i in range(X_train_const.shape[1])]

pd.DataFrame(zip(X_train_const.columns,vif),columns=['Col Names','VIF Values'])
model_ols=sm.OLS(y_train_rfe5,X_train_const).fit()

model_ols.summary()
#Checking the Adjusted R square value

print(round((model_ols.rsquared)*100,2))
print('The parameters which the model learns is\n',model_ols.params)
sgd=SGDRegressor(penalty=None,alpha=0.001)
model_sgd=sgd.fit(X_train_rfe5,y_train_rfe5)
model_sgd.intercept_
list(zip(X_train_rfe5,model_sgd.coef_))
poly=PolynomialFeatures(degree=2)
X_poly=poly.fit_transform(X_rfe5)

y=y

X_train_poly,X_test_poly,y_train_poly,y_test_poly=train_test_split(X_poly,y,random_state=45)
model_poly=lr.fit(X_train_poly,y_train_poly)

print(f'The R square for train data is {round(model_poly.score(X_train_poly,y_train_poly)*100,2)}')

print(f'The R square for test data is {round(model_poly.score(X_test_poly,y_test_poly)*100,2)}')

print(f'The MSE for train data is {mean_squared_error(y_train_poly,model_poly.predict(X_train_poly))}')

print(f'The MSE for test data is {mean_squared_error(y_test_poly,model_poly.predict(X_test_poly))}')
rf=RandomForestRegressor(n_estimators=100,min_samples_leaf=5,max_depth=20)
X_random=X_rfe5

y=y



X_train_random,X_test_random,y_train_random,y_test_random=train_test_split(X_random,y,random_state=45)
model_random=rf.fit(X_train_random,y_train_random)



print(f'The R square for train data is {round(model_random.score(X_train_random,y_train_random)*100,2)}')

print(f'The R square for test data is {round(model_random.score(X_test_random,y_test_random)*100,2)}')

print(f'The MSE for train data is {mean_squared_error(y_train_random,model_random.predict(X_train_random))}')

print(f'The MSE for test data is {mean_squared_error(y_test_random,model_random.predict(X_test_random))}')
print(f'Linear regression mean R square value is {round(cross_val_score(model_linear5,X_rfe5,y,cv=10).mean()*100,2)}')

print(f'Polynomial regression mean R square value is {round(cross_val_score(model_poly,X_poly,y,cv=10).mean()*100,2)}')

print(f'random Forest regression mean R square value is {round(cross_val_score(model_random,X_random,y,cv=10).mean()*100,2)}')

print(f'Gradient Descent mean R square value is {round(cross_val_score(model_sgd,X_rfe5,y,cv=10).mean()*100,2)}')