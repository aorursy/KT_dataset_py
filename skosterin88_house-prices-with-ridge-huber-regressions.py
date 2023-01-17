%matplotlib inline

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.shape
df_test.shape
df_train.columns
df_test.columns
df_train.head()
df_test.head()
df_train.info()
df_train['MSSubClass'] = df_train['MSSubClass'].astype('str')

df_train['OverallQual'] = df_train['OverallQual'].astype('str')

df_train['OverallCond'] = df_train['OverallCond'].astype('str')

df_train['MoSold'] = df_train['MoSold'].astype('str')
df_train.info()
df_train_cat = df_train[df_train.select_dtypes(include=['object']).columns]
df_train_num = df_train[df_train.select_dtypes(exclude=['object']).columns[1:].drop('SalePrice')]
sns.distplot(df_train['SalePrice'])



fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'])



fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
y = df_train['SalePrice']
df_train_num = df_train_num.fillna(df_train_num.median())
skewness_num = df_train_num.apply(lambda x: skew(x))

skewness_num.sort_values(ascending=False)
skewness_num = skewness_num[abs(skewness_num)>0.5]

skewness_num.index
df_train_num = df_train_num[skewness_num.index].apply(lambda x: np.log(x+1))
df_train_num.head()
df_train_cat = pd.get_dummies(df_train_cat)
df_train_cat.shape
df_train_cat.head()
df_train = pd.concat([df_train_num,df_train_cat], axis=1)
df_train.head()
df_train.describe()
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, HuberRegressor

from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.3, random_state=17)
def rmse_CV_train(model):

    rmse = np.sqrt(-cross_val_score(model,X_train,y_train,scoring ="neg_mean_squared_error",cv=5))

    return (rmse)

def rmse_CV_test(model):

    rmse = np.sqrt(-cross_val_score(model,X_test,y_test,scoring ="neg_mean_squared_error",cv=5))

    return (rmse)
lr = LinearRegression()

lr.fit(X_train, y_train)

test_pred = lr.predict(X_test)

train_pred = lr.predict(X_train)



huber = HuberRegressor()

huber.fit(X_train,y_train)

test_pred_huber = huber.predict(X_test)

train_pred_huber = huber.predict(X_train)

print("Linear regression RMSE on train and test: ", [rmse_CV_train(lr).mean(), rmse_CV_test(lr).mean()])

print("Huber regression RMSE on train and test: ", [rmse_CV_train(huber).mean(), rmse_CV_test(huber).mean()])
plt.scatter(train_pred, train_pred - y_train, c = "blue",  label = "Training data (Linear)")

plt.scatter(test_pred,test_pred - y_test, c = "black",  label = "Validation data (Linear)")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



plt.scatter(train_pred_huber, train_pred_huber - y_train, c = "blue",  label = "Training data (Huber)")

plt.scatter(test_pred_huber,test_pred_huber - y_test, c = "black",  label = "Validation data (Huber)")

plt.title("Huber regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
plt.scatter(train_pred, y_train, c = "blue",  label = "Training data (Linear)")

plt.scatter(test_pred, y_test, c = "black",  label = "Validation data (Linear)")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()



plt.scatter(train_pred_huber, y_train, c = "blue",  label = "Training data (Huber)")

plt.scatter(test_pred_huber, y_test, c = "black",  label = "Validation data (Huber)")

plt.title("Huber regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

ridge.fit(X_train,y_train)

alpha = ridge.alpha_

print('best alpha',alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = 5)

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)

print("Ridge RMSE on Training set :", rmse_CV_train(ridge).mean())

print("Ridge RMSE on Test set :", rmse_CV_test(ridge).mean())

y_train_rdg = ridge.predict(X_train)

y_test_rdg = ridge.predict(X_test)
plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "blue",  label = "Training data")

plt.scatter(y_test_rdg, y_test_rdg - y_test, c = "black", marker = "v", label = "Validation data")

plt.title("Linear regression with Ridge regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
plt.scatter(y_train_rdg, y_train, c = "blue",  label = "Training data")

plt.scatter(y_test_rdg, y_test, c = "black",  label = "Validation data")

plt.title("Linear regression with Ridge regularization")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()
df_test['MSSubClass'] = df_test['MSSubClass'].astype('str')

df_test['OverallQual'] = df_test['OverallQual'].astype('str')

df_test['OverallCond'] = df_test['OverallCond'].astype('str')

df_test['MoSold'] = df_test['MoSold'].astype('str')
df_test_cat = df_test[df_test.select_dtypes(include=['object']).columns]
df_test_num = df_test[df_test.select_dtypes(exclude=['object']).columns[1:]]
df_test_num = df_test_num.fillna(df_test_num.median())
df_test_num = df_test_num[skewness_num.index].apply(lambda x: np.log(x+1))

df_test_num.head(15)
df_test_cat = pd.get_dummies(df_train_cat)
df_test_cat.shape
df_test_cat.head()
df_test = pd.concat([df_test_num,df_test_cat], axis=1)
X_train.shape, df_test.shape
df_test = df_test.dropna()
y_valid_ridge = ridge.predict(df_test)

y_valid_huber = huber.predict(df_test)
y_valid_ridge = pd.DataFrame(np.exp(y_valid_ridge), index=np.arange(1461,1461+df_test.shape[0]),columns=['SalePrice'])

y_valid_huber = pd.DataFrame(np.exp(y_valid_huber), index=np.arange(1461,1461+df_test.shape[0]),columns=['SalePrice'])
y_valid_ridge.head()
y_valid_huber.head()
def create_submission_file(out_file, data, init_index=1461, target='SalePrice', index_label='Id'):

    df_submit = pd.DataFrame(data,

                                 index=np.arange(init_index,init_index+data.shape[0]),

                                 columns=[target])

    data.to_csv(out_file, index_label=index_label)
create_submission_file('houses_ridgeregression.csv', y_valid_ridge)

create_submission_file('houses_huberregression.csv', y_valid_huber)