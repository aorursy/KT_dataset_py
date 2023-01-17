#imports

import numpy as np

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
traffic_json = pd.read_json('../input/traffic/traffic_rules.json')

traffic_csv = traffic_json.to_csv()

print(traffic_csv)
traffic_xlsx = pd.read_excel('../input/traffic/traffic_rules.xlsx', sheet_name = None, index_col = 0)

traffic_rules = traffic_xlsx['rules']

traffic_rules
traffic_categorical = traffic_xlsx['rules_categorical']

traffic_categorical
traffic_ordinal = traffic_xlsx['rules_ordinal']

traffic_ordinal
X = traffic_ordinal.drop(columns = 'Pass').values

y = traffic_ordinal['Pass'].values

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size = 0.6, test_size=0.4, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest,train_size = 0.6, test_size=0.4, random_state=42)

lr = LinearRegression()

lr.fit(X_train,y_train)
print('MSE on validation data is: ', mean_squared_error(y_val,lr.predict(X_val)))

print('MSE on test data is: ', mean_squared_error(y_test,lr.predict(X_test)))
X = traffic_categorical.drop(columns = 'Pass').values

y = traffic_categorical['Pass'].values

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size = 0.6, test_size=0.4, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest,train_size = 0.6, test_size=0.4, random_state=42)

lr = LinearRegression()

lr.fit(X_train,y_train)
print('MSE on validation data is: ', mean_squared_error(y_val,lr.predict(X_val)))

print('MSE on test data is: ', mean_squared_error(y_test,lr.predict(X_test)))