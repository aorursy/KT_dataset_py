import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
train = pd.read_csv("/kaggle/input/usedcarprices/Data_Train.csv")
test = pd.read_csv("/kaggle/input/usedcarprices/Data_Test.csv")
train.head()
train.shape
train.info()
train.nunique()
train.isnull().sum()
train.shape
train.Fuel_Type.value_counts()
train = train[train['Fuel_Type'] != 'Electric']
train.shape
print(train.Kilometers_Driven.min())
print(train.Kilometers_Driven.max())
plt.boxplot(train.Kilometers_Driven)
print(len(train[train['Kilometers_Driven'] > 500000]))
print(len(train[train['Kilometers_Driven'] < 1000]))
train = train[train['Kilometers_Driven'] < 500000]
train = train[train['Kilometers_Driven'] > 1000]
train.shape

train.Name = train.Name.str.split().str.get(0)
test.Name = test.Name.str.split().str.get(0)
train.head()
train.Name.value_counts()
train = train[train['Name'] != 'Force']
train = train[train['Name'] != 'ISUZU']
train = train[train['Name'] != 'Bentley']
train = train[train['Name'] != 'Lamborghini']
train = train[train['Name'] != 'Isuzu']
train = train[train['Name'] != 'Smart']
train = train[train['Name'] != 'Ambassador']
train.shape
sns.boxplot(train.Price)
print(train.Price.min())
print(train.Price.max())
train = train[train.Price < 120]
train.shape
train = train[train.Price > 0.5]
train.shape
train.Mileage = train.Mileage.str.split().str.get(0).astype('float')
train.Engine = train.Engine.str.split().str.get(0).astype('int', errors='ignore')
train.Power = train.Power.str.split().str.get(0).astype('float', errors='ignore')
train.head()

test.Mileage = test.Mileage.str.split().str.get(0).astype('float')
test.Engine = test.Engine.str.split().str.get(0).astype('int', errors='ignore')
test.Power = test.Power.str.split().str.get(0).astype('float', errors='ignore')
train['Car_age'] = 2020 - train['Year']
test['Car_age'] = 2020 - test['Year']
train.head()
train.Price = np.log1p(train.Price)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train['Name'] = label_encoder.fit_transform(train['Name'])
train['Location'] = label_encoder.fit_transform(train['Location'])
train['Fuel_Type'] = label_encoder.fit_transform(train['Fuel_Type'])
train['Transmission'] = label_encoder.fit_transform(train['Transmission'])
train['Owner_Type'] = label_encoder.fit_transform(train['Owner_Type'])


test['Name'] = label_encoder.fit_transform(test['Name'])
test['Location'] = label_encoder.fit_transform(test['Location'])
test['Fuel_Type'] = label_encoder.fit_transform(test['Fuel_Type'])
test['Transmission'] = label_encoder.fit_transform(test['Transmission'])
test['Owner_Type'] = label_encoder.fit_transform(test['Owner_Type'])

train.head()
train.isnull().sum()
train.dtypes
train.Engine = pd.to_numeric(train.Engine, errors='coerce')
train.Power = pd.to_numeric(train.Power, errors='coerce')
test.Engine = pd.to_numeric(test.Engine, errors='coerce')
test.Power = pd.to_numeric(test.Power, errors='coerce')
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
train[["Engine", "Power", "Seats"]] = imputer.fit_transform(train[["Engine", "Power", "Seats"]])
test[["Engine", "Power", "Seats"]] = imputer.fit_transform(test[["Engine", "Power", "Seats"]])

train.isnull().sum()
y = train.Price
X = train.drop(['Price'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LinearRegression

model1 = LinearRegression() 
model1.fit(X_train, y_train) 

y_pred = model1.predict(X_valid) 
from sklearn import metrics
from sklearn.metrics import r2_score

print('Mean Absolute Error:', metrics.mean_absolute_error(y_valid, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_valid, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_valid, y_pred)))
print("R2 score : %f" % r2_score(y_valid,y_pred))
from sklearn.ensemble import RandomForestRegressor

model2 = RandomForestRegressor(n_estimators=200)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_valid)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_valid, y_pred2))
print('Mean Squared Error:', metrics.mean_squared_error(y_valid, y_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_valid, y_pred2)))
print("R2 score : %f" % r2_score(y_valid,y_pred2))
from xgboost import XGBRegressor

model3 = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model3.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)
y_pred3 = model3.predict(X_valid)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_valid, y_pred3))
print('Mean Squared Error:', metrics.mean_squared_error(y_valid, y_pred3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_valid, y_pred3)))
print("R2 score : %f" % r2_score(y_valid,y_pred3))
from sklearn.linear_model import Ridge

model4 = Ridge(alpha=1.0)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_valid)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_valid, y_pred4))
print('Mean Squared Error:', metrics.mean_squared_error(y_valid, y_pred4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_valid, y_pred4)))
print("R2 score : %f" % r2_score(y_valid,y_pred4))
from sklearn.linear_model import Lasso

model5 = Lasso(alpha=1.0)
model5.fit(X_train, y_train)
y_pred5 = model5.predict(X_valid)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_valid, y_pred5))
print('Mean Squared Error:', metrics.mean_squared_error(y_valid, y_pred5))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_valid, y_pred5)))
print("R2 score : %f" % r2_score(y_valid,y_pred5))
final_test_predictions = model3.predict(test)
final_test_predictions = np.exp(final_test_predictions)-1 #converting target to original state
type(final_test_predictions)
test['Price'] = pd.Series(final_test_predictions)
test.to_csv('predictions.csv', index=False)