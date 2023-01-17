

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import sklearn

types_dict_train = {'train_id': 'int64',

             'item_condition_id': 'int8',

             'price': 'float64',

             'shipping': 'int8'}



train_df = pd.read_csv('../input/train.tsv',delimiter='\t',low_memory=True,dtype=types_dict_train)

types_dict_test = {'test_id': 'int64',

             'item_condition_id': 'int8',

             'shipping': 'int8'}

test_df = pd.read_csv('../input/test.tsv',delimiter='\t',low_memory= True,dtype=types_dict_test)
train = pd.read_table('../input/train.tsv')
# first look with training set 

train_df = train_df.rename(columns = {'train_id' : 'id'})

train_df.head()
# and with test set

test_df = test_df.rename(columns = {"test_id" : "id"})

test_df.head()


plt.figure(figsize=(20,15))

plt.hist(train_df['price'] , bins = 50 , range = [0,300] , label = 'price')

plt.xlabel('Price')

plt.ylabel('Sample')

plt.title('Sale Price Distribution')

plt.show()
train_df['price'].describe()
print(train_df.isnull().sum() , train_df.isnull().sum()/train_df.shape[0] * 100)
train_df['is_train'] = 1

test_df['is_train'] = 0 
train_test_combine = pd.concat([train_df.drop(['price'] , axis = 1) , test_df], axis = 0)
train_test_combine.head()


train_test_combine.category_name = train_test_combine.category_name.astype('category')

train_test_combine.item_description = train_test_combine.item_description.astype('category')

train_test_combine.name = train_test_combine.name.astype('category')

train_test_combine.brand_name = train_test_combine.brand_name.astype('category')

train_test_combine.name = train_test_combine.name.cat.codes

train_test_combine.brand_name = train_test_combine.brand_name.cat.codes

train_test_combine.item_description = train_test_combine.item_description.cat.codes

train_test_combine.category_name = train_test_combine.category_name.cat.codes
# show train_test_combine

train_test_combine.head()
#drop colums brand_name

train_test_combine = train_test_combine.drop(['brand_name'] , axis = 1)
train_test_combine['category_name'] = train_test_combine['category_name'].fillna(train_test_combine['category_name'].mode())

train_test_combine['item_description'] = train_test_combine['item_description'].fillna(train_test_combine['item_description'].mode())


train_df = train_test_combine.loc[train_test_combine['is_train'] == 1]

test_df = train_test_combine.loc[train_test_combine['is_train'] == 0]
#drop colum a is_train in train_df and test_df

train_df = train_df.drop(['is_train'] , axis = 1)

test_df = test_df.drop(['is_train'] , axis = 1)
train_df.head()
test_df.head()
#Create feature and target to training with random forest

train_df['price'] = train.price

features_rdf , target_rdf = train_df.drop(['price'] , axis = 1) , train_df.price
train_df['price'] = train_df['price'].apply(lambda x: np.log(x) if x > 0 else x)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



random_forest_model = RandomForestRegressor(n_jobs = -1, min_samples_leaf = 3 , n_estimators = 200)

random_forest_model.fit(features_rdf , target_rdf)

random_forest_model.score(features_rdf , target_rdf)
#Predict and submission

predict_df = random_forest_model.predict(test_df)

predict_df = pd.Series(np.exp(predict_df))

submission = pd.concat([test_df.id , predict_df] , axis = 1)

submission.columns = ['test_id' , 'price']

submission.to_csv('./rdf_kernel_1.csv' , index = False)