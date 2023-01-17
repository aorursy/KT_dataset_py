import pandas as pd

import numpy as np



pd.options.display.max_columns = 999

pd.options.display.max_colwidth = 999



# ignore warnings

import warnings

warnings.filterwarnings(action='ignore')
# read in data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print ("Train data shape:", train.shape)

print ("Test data shape:", test.shape)
train.head()
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 10)
train.SalePrice.describe()
print ("Skew is:", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.show()
target = np.log(train.SalePrice)

print ("Skew is:", target.skew())

plt.hist(target, color='blue')

plt.show()
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.dtypes
corr = numeric_features.corr()



print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')

print (corr['SalePrice'].sort_values(ascending=False)[-5:])
train.OverallQual.unique()
quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
quality_pivot
quality_pivot.plot(kind='bar', color='blue')

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
plt.scatter(x=train['GrLivArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
plt.scatter(x=train['GarageArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
train = train[train['GarageArea'] < 1200]
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls
print ("Unique values are:", train.MiscFeature.unique())
categoricals = train.select_dtypes(exclude=[np.number])
cat_vars = categoricals.columns
categoricals.describe()
print ("Original: \n") 

print (train.Street.value_counts(), "\n")
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
print ('Encoded: \n') 

print (train.enc_street.value_counts())
condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='blue')

plt.xlabel('Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
train['enc_condition'] = train.SaleCondition.apply(lambda x: 1 if x=='Partial' else 0)

test['enc_condition'] = test.SaleCondition.apply(lambda x: 1 if x=='Partial' else 0)
condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='blue')

plt.xlabel('Encoded Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
# I'm going to factorize all the categoricals 



print (train.select_dtypes(include=[np.number]).shape)

for cat in cat_vars:

    train[cat] = train[cat].factorize()[0]

print (train.select_dtypes(include=[np.number]).shape)



print (test.select_dtypes(include=[np.number]).shape)

for cat in cat_vars:

    test[cat] = test[cat].factorize()[0]

    

print (test.select_dtypes(include=[np.number]).shape)
data = train.select_dtypes(include=[np.number]).interpolate()
sum(data.isnull().sum() != 0)
y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
from sklearn import linear_model

from sklearn.ensemble import GradientBoostingRegressor

lr = GradientBoostingRegressor()
model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, predictions))
actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.75, color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()
rm = linear_model.Ridge(alpha=.030)
ridge_model = rm.fit(X_train, y_train)

preds_ridge = ridge_model.predict(X_test)
# score

print ("R^2 is: \n", ridge_model.score(X_test, y_test))

print ('RMSE is: \n', mean_squared_error(y_test, predictions))
plt.scatter(preds_ridge, actual_values, alpha=.75, color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Ridge Regularization')

plt.show()
for i in range (-2, 3):

    alpha = 10**i

    rm = linear_model.Ridge(alpha=alpha)

    ridge_model = rm.fit(X_train, y_train)

    preds_ridge = ridge_model.predict(X_test)

    #compute r score

    print ('alpha is: ', alpha )

    print ("R^2 is: \n", ridge_model.score(X_test, y_test)) #compute the R^2 score

    print ('RMSE is: \n', mean_squared_error(y_test, predictions))

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b') #alpha helps to show overlapping data

    plt.xlabel('Predicted Price')

    plt.ylabel('Actual Price')

    plt.title('Ridge Regularization')

    plt.show()
submission = pd.DataFrame()
submission['Id'] = test.Id
feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
final_predictions = np.exp(predictions)
print ("Original predictions are:", predictions[:5], "\n")

print ("Final predictions are:", final_predictions[:5])
submission['SalePrice'] = final_predictions
submission.head() #check that everything looks good
submission.to_csv('submission1.csv', index=False) # prevents pandas from reindexing 
import pandas as pd

import numpy as np



pd.options.display.max_columns = 999

pd.options.display.max_colwidth = 999