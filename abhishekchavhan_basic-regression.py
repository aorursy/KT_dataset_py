import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,BaggingRegressor
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
for label,content in train_data.items():
    if pd.api.types.is_numeric_dtype(content):
        train_data[label] = content.fillna(content.median())
    if not pd.api.types.is_numeric_dtype(content):
        train_data[label] = pd.Categorical(content).codes + 1
X = train_data.drop('SalePrice',axis=1)
y = train_data['SalePrice']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
model = RandomForestRegressor()
model.fit(X_train,y_train)
model.score(X_test,y_test)
model_1 = GradientBoostingRegressor()
model_1.fit(X_train,y_train)
model_1.score(X_test,y_test)
model_2 = ExtraTreesRegressor()
model_2.fit(X_train,y_train)
model_2.score(X_test,y_test)
model_3 = BaggingRegressor()
model_3.fit(X_train,y_train)
model_3.score(X_test,y_test)
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
for label,content in test_data.items():
    if pd.api.types.is_numeric_dtype(content):
        test_data[label] = content.fillna(content.median())
    if not pd.api.types.is_numeric_dtype(content):
        test_data[label] = pd.Categorical(content).codes + 1
y_preds = model_1.predict(test_data)
Sub = pd.DataFrame()

Sub['Id'] = test_data['Id']
Sub['SalePrice'] = y_preds
Sub
Sub.to_csv('Submission1908.csv',index=False)
