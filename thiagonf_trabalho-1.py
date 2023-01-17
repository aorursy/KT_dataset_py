import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

file_train = '../input/train.csv'

file_valid = '../input/valid.csv'

file_test = '../input/test.csv'

file_exemplo = '../input/exemplo.csv'

data_train = pd.read_csv(file_train)

data_valid = pd.read_csv(file_valid)

data_test = pd.read_csv(file_test)

data_exemplo = pd.read_csv(file_exemplo)

print(len(data_train), len(data_valid), len(data_test))
Y = data_train['sale_price']

ID_predict = data_exemplo['sale_id']

data_train = data_train.drop('sale_price', axis=1)



data_all = data_train.append(data_valid).append(data_test)

data_all = data_all.drop(['address', 'ease-ment', 'sale_id'], axis=1)

len(data_all)
cols=[i for i in data_all.columns if i not in ['building_class_category','sale_date','tax_class_at_present','building_class_at_present',

                                               'building_class_at_time_of_sale', 'tax_class_at_time_of_sale', 'neighborhood', 

                                               'apartment_number']]

for col in cols:

    data_all[col]=pd.to_numeric(data_all[col], errors='coerce')



data_all.info()
null_columns=data_all.columns[data_all.isnull().any()]

data_all[null_columns].isnull().sum()
data_all['land_square_feet'].fillna(data_all['land_square_feet'].mean(), inplace=True)

data_all['gross_square_feet'].fillna(data_all['gross_square_feet'].mean(), inplace=True)

data_all['sale_date'] = pd.to_datetime(data_all.sale_date, format='%m/%d/%y').astype(int)

null_columns=data_all.columns[data_all.isnull().any()]

data_all[null_columns].isnull().sum()
list_quantitative = [f for f in data_all.columns if data_all.dtypes[f] != 'object']

list_qualitative = [f for f in data_all.columns if data_all.dtypes[f] == 'object']



print("QUANT", list_quantitative)

print("QUALI", list_qualitative)
data_all = pd.concat([data_all, pd.get_dummies(data_all['building_class_category'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['tax_class_at_present'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['building_class_at_present'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['tax_class_at_time_of_sale'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['building_class_at_time_of_sale'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['neighborhood'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['apartment_number'])], axis=1);



data_all = data_all.drop(['building_class_category', 'tax_class_at_present', 'building_class_at_present', 

                          'tax_class_at_time_of_sale', 'building_class_at_time_of_sale', 'neighborhood', 'apartment_number'], axis=1)
from sklearn.model_selection import train_test_split

# Create X

data_train = data_all[:(len(data_train))]

data_valid = data_all[(len(data_train)):((len(data_train))+(len(data_valid)))]

data_test = data_all[((len(data_train))+(len(data_valid))):]



print()



X = data_train

print(len(data_train), len(data_valid), len(data_test))

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_leaf_nodes=100)

model.fit(X,Y)
# from sklearn.linear_model import LinearRegression

# model = LinearRegression()

# model.fit(X, Y)
# from sklearn.ensemble import RandomForestRegressor



# model = RandomForestRegressor(n_estimators=100, bootstrap=False, n_jobs=-1, max_leaf_nodes=100)

# model.fit(X, Y)

# print(model.score(X,Y))
data_predict = data_valid.append(data_test)

Y_predict = model.predict(data_predict)



Y_predict
data_to_submit = pd.DataFrame({

    'sale_id': ID_predict,

    'sale_price':Y_predict

})



data_to_submit.to_csv('csv_to_submit.csv', index = False)