import pandas as pd

from sklearn.tree import DecisionTreeRegressor

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt

import seaborn as sns



import os



train_data = pd.read_csv("../input/train.csv")

valid_data = pd.read_csv("../input/valid.csv")

test_data = pd.read_csv("../input/test.csv") 



train_data.drop(['sale_date', 'neighborhood', 'building_class_category', 'ease-ment', 'building_class_at_present', 

                 'address', 'apartment_number', 'building_class_at_time_of_sale'], axis=1, inplace=True) 



valid_data.drop(['sale_date', 'neighborhood', 'building_class_category', 'ease-ment', 'building_class_at_present',

                 'address', 'apartment_number', 'building_class_at_time_of_sale'], axis=1, inplace=True) 



test_data.drop(['sale_date', 'neighborhood', 'building_class_category', 'ease-ment', 'building_class_at_present', 

                'address', 'apartment_number', 'building_class_at_time_of_sale'], axis=1, inplace=True) 



train_data['land_square_feet'] = pd.to_numeric(train_data.land_square_feet, errors='coerce')

train_data['land_square_feet'].fillna(train_data["land_square_feet"].mean(), inplace=True)

train_data.loc[train_data.land_square_feet== 0, 'land_square_feet'] = train_data.land_square_feet.mean()

valid_data['land_square_feet'] = pd.to_numeric(valid_data.land_square_feet, errors='coerce')

valid_data['land_square_feet'].fillna(valid_data["land_square_feet"].mean(), inplace=True)

valid_data.loc[valid_data.land_square_feet== 0, 'land_square_feet'] = valid_data.land_square_feet.mean()

test_data['land_square_feet'] = pd.to_numeric(test_data.land_square_feet, errors='coerce')

test_data['land_square_feet'].fillna(test_data["land_square_feet"].mean(), inplace=True)

test_data.loc[test_data.land_square_feet== 0, 'land_square_feet'] = test_data.land_square_feet.mean()



train_data['gross_square_feet'] = pd.to_numeric(train_data.gross_square_feet, errors='coerce')

train_data['gross_square_feet'].fillna(train_data["gross_square_feet"].mean(), inplace=True)

train_data.loc[train_data.gross_square_feet== 0, 'gross_square_feet'] = train_data.gross_square_feet.mean()

valid_data['gross_square_feet'] = pd.to_numeric(valid_data.gross_square_feet, errors='coerce')

valid_data['gross_square_feet'].fillna(valid_data["gross_square_feet"].mean(), inplace=True)

valid_data.loc[valid_data.gross_square_feet== 0, 'gross_square_feet'] = valid_data.gross_square_feet.mean()

test_data['gross_square_feet'] = pd.to_numeric(test_data.gross_square_feet, errors='coerce')

test_data['gross_square_feet'].fillna(test_data["gross_square_feet"].mean(), inplace=True)

test_data.loc[test_data.gross_square_feet== 0, 'gross_square_feet'] = test_data.gross_square_feet.mean()



train_data.loc[train_data.year_built== 0, 'year_built'] = train_data.year_built.mean()

valid_data.loc[valid_data.year_built== 0, 'year_built'] = valid_data.year_built.mean()

test_data.loc[test_data.year_built== 0, 'year_built'] = test_data.year_built.mean()



new_train_data = pd.get_dummies(train_data)

new_valid_data = pd.get_dummies(valid_data)

new_test_data = pd.get_dummies(test_data)   



new_train_data.drop(['sale_id'], axis=1, inplace=True)

new_valid_data.drop(['sale_id'], axis=1, inplace=True)

new_test_data.drop(['sale_id'], axis=1, inplace=True)



target = new_train_data['sale_price']

features = new_train_data.drop('sale_price', axis=1)



tree = DecisionTreeRegressor()

tree = tree.fit(features, target)



# print(tree.score(features, target))



valid = pd.DataFrame()

valid['sale_id'] = valid_data['sale_id']

valid['sale_price'] = tree.predict(new_valid_data)



test = pd.DataFrame()

test['sale_id'] =  test_data['sale_id']

test['sale_price'] =  tree.predict(new_test_data)



submission = pd.DataFrame()

submission = pd.concat([valid, test])



submission.to_csv('submission.csv', index=False)

submission.shape