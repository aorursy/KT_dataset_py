import pandas as pd

import numpy as np
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.describe()
train['MSSubClass'].describe()
train['MSZoning'].describe()
train['LotFrontage'].describe()

train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())

train['LotFrontage'].describe()

train['MasVnrArea'].describe()

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].median())
train['MasVnrArea'].describe()
train.head()
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10,6)

print(train['SalePrice'].describe())

print('skew is ', train.SalePrice.skew())

plt.hist(train.SalePrice,color = 'green')

plt.show()
target = np.log(train['SalePrice'])

print('skew is = ', target.skew())

plt.hist(target, color = 'green')

plt.show()

numeric_features = train.select_dtypes(include=[np.number])

numeric_features.dtypes
corr = numeric_features.corr()



print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')

print(corr['SalePrice'].sort_values(ascending = False)[-5:])
plt.scatter(x = train['GrLivArea'], y = target)

plt.ylabel('salePrice')

plt.xlabel('Above ground')

plt.show()
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls
categoricals = train.select_dtypes(exclude=[np.number])

categoricals.describe()
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
data = train.select_dtypes(include=[np.number]).interpolate().dropna() 

y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)
sum(data.isnull().sum() != 0)
y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model

lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, predictions))
actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.75,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()
for i in range (-2, 3):

    alpha = 10**i

    rm = linear_model.Ridge(alpha=alpha)

    ridge_model = rm.fit(X_train, y_train)

    preds_ridge = ridge_model.predict(X_test)

    

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')

    plt.xlabel('Predicted Price')

    plt.ylabel('Actual Price')

    plt.title('Ridge Regularization with alpha = {}'.format(alpha))

    overlay = 'R^2 is: {}\nRMSE is: {}'.format(

                    ridge_model.score(X_test, y_test),

                    mean_squared_error(y_test, preds_ridge))

    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')

    plt.show()
submission = pd.DataFrame()

submission['Id'] = test.Id
feats = test.select_dtypes(

        include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
final_predictions = np.exp(predictions)

submission['SalePrice'] = final_predictions

submission.head()
submission.to_csv('submission1.csv', index=False)
# I Took the Tutorial On data quest. on that basis i did these, now i am planning to do it myself. 

# Any feedbacks and suggestions are most welcome.