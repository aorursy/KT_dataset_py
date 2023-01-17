#Moroz Roman
import pandas as pd                 

import numpy as np

import matplotlib.pyplot as plt     

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import linear_model
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print("Train data shape:", train.shape)

print("Test data shape:", test.shape)
plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)
print ("Skew is:", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.show()
target = np.log(train.SalePrice)

print ("\n Skew is:", target.skew())

plt.hist(target, color='blue')

plt.show()
numeric_features = train.select_dtypes(include=[np.number])

print(numeric_features.dtypes)
corr = numeric_features.corr()

print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')

print (corr['SalePrice'].sort_values(ascending=False)[-5:])
print(train.OverallQual.unique())
quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)

print(quality_pivot)
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

plt.xlim(-200,1600)     # This forces the same scale as before

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'



print(nulls)
print ("Unique values are:", train.MiscFeature.unique())
categoricals = train.select_dtypes(exclude=[np.number])

#categoricals.describe()

print(categoricals.describe())
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)



print ('Encoded: \n')

print (train.enc_street.value_counts())
condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='blue')

plt.xlabel('Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
def encode(x): return 1 if x == 'Partial' else 0

train['enc_condition'] = train.SaleCondition.apply(encode)

test['enc_condition'] = test.SaleCondition.apply(encode)
condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='blue')

plt.xlabel('Encoded Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

print(sum(data.isnull().sum() != 0))
y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)
predictions = model.predict(X_test)
actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.75,

            color='b')  # alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()


submission = pd.DataFrame()

submission['Id'] = test.Id
feats = test.select_dtypes(

    include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)

final_predictions = np.exp(predictions)
submission['SalePrice'] = final_predictions

print(submission.head())
submission.to_csv('submission1.csv', index=False)