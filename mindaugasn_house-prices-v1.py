import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn import metrics
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.tail()
test.head()
train.info()
plt.figure(figsize=(16, 6))

sns.heatmap(data=train.isnull(), cmap='viridis', yticklabels=False, cbar=False)
plt.figure(figsize=(16, 6))

sns.heatmap(data=test.isnull(), cmap='viridis', yticklabels=False, cbar=False)
train.describe()
plt.figure(figsize=(16, 12))

sns.heatmap(train.corr(), cmap='coolwarm')
correlation_data = train.corr()
best_corr = correlation_data[(correlation_data['SalePrice'] > 0.5) | (correlation_data['SalePrice'] < -0.5)]
plt.figure(figsize=(15, 12))

sns.heatmap(best_corr, cmap='coolwarm')
best_list = list(best_corr.index)

training_columns = best_list[:]

training_columns.append('Id')

testing_columns = best_list[:-1]

# testing_columns.append('Id')
train_for_model = train[training_columns]
sns.pairplot(train_for_model.drop('Id', axis=1))
sns.distplot(train_for_model['SalePrice'])
X = train_for_model.drop(['Id', 'SalePrice'], axis=1)

y = train_for_model['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
def return_metrics(predictions):

    #evaluation metrics

    MAE = metrics.mean_absolute_error(y_test, predictions)

    MSE = metrics.mean_squared_error(y_test, predictions)

    MRSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))

    print('MAE is {}'.format(MAE))

    print('MSE is {}'.format(MSE))

    print('MRSE is {}'.format(MRSE))
def model_predictions(model):

    

    # Fitting model

    my_model = model()

    my_model.fit(X_train, y_train)

    predictions = my_model.predict(X_test)

    

    # Plotting predictions

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))



    axes[0].scatter(y_test, predictions)

    axes[1].hist((y_test - predictions), bins=30)

    plt.tight_layout()

    plt.show()

    

    # Return evaluation metrics

    return_metrics(predictions)

    

    # Try to print coefficients

    try:

        cdf = pd.DataFrame(data=my_model.coef_, index=X.columns, columns=['Coeff'])

        print('\n')

        print(cdf)

    except:

        print('\n')

        print('Model does not have coefficients')
model_predictions(LinearRegression)
model_predictions(RandomForestRegressor)
test[testing_columns].head()
test[testing_columns].info()
test[test['GarageCars'].isnull()]['GarageCars']
test[test['GarageArea'].isnull()]['GarageArea']
test[test['TotalBsmtSF'].isnull()]['TotalBsmtSF']
test['GarageCars'] = test['GarageCars'].iloc[1116] = 0

test['GarageArea'] = test['GarageArea'].iloc[1116] = 0

test['TotalBsmtSF'] = test['TotalBsmtSF'].iloc[660] = 0
my_model = RandomForestRegressor()

my_model.fit(X, y)

predictions = my_model.predict(test[testing_columns])

submission = pd.DataFrame({

    'Id': test['Id'],

    'SalePrice': predictions

}) 



submission.to_csv('submission.csv', index=False)