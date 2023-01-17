#simple-mercari-price-code



#read

import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt



train = pd.read_table("../input/train.tsv")

test = pd.read_table("../input/test_stg2.tsv")

pd.set_option('display.max_columns', 10)



print(train.shape)

print(test.shape)

print(train.columns)



#copy

train_veiw = train

train_veiw.category_name = train_veiw.category_name.astype('category')

train_veiw.item_description = train_veiw.item_description.astype('category')

train_veiw.name = train_veiw.name.astype('category')

train_veiw.brand_name = train_veiw.brand_name.astype('category')





train_veiw.category_name = train_veiw.category_name.cat.codes

train_veiw.item_description = train_veiw.item_description.cat.codes

train_veiw.name = train_veiw.name.cat.codes

train_veiw.brand_name = train_veiw.brand_name.cat.codes



train_veiw.hist(bins = 50, figsize = (15,10))

#pre-processing



train = train.rename(columns = {'train_id':'id'})

test = test.rename(columns = {'test_id':'id'})



train['is_train'] = 1

test['is_train'] = 0



train_test_combine = pd.concat([train.drop(['price'], axis=1), test], axis=0)



train_test_combine.category_name = train_test_combine.category_name.astype('category')

train_test_combine.item_description = train_test_combine.item_description.astype('category')

train_test_combine.name = train_test_combine.name.astype('category')

train_test_combine.brand_name = train_test_combine.brand_name.astype('category')



train_test_combine.category_name = train_test_combine.category_name.cat.codes

train_test_combine.item_description = train_test_combine.item_description.cat.codes

train_test_combine.name = train_test_combine.name.cat.codes

train_test_combine.brand_name = train_test_combine.brand_name.cat.codes



df_test = train_test_combine.loc[train_test_combine['is_train'] == 0]

df_train = train_test_combine.loc[train_test_combine['is_train'] == 1]



df_train = df_train.drop(['is_train'], axis=1)

df_test = df_test.drop(['is_train'], axis=1)



df_train['price'] = train.price

df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x>0 else x)



print(train.head())

print(df_train.head())
#Model



from sklearn.ensemble import RandomForestRegressor



x_train = df_train.drop(['price'], axis = 1)

y_train = df_train.price



model = RandomForestRegressor(n_jobs = -1, min_samples_leaf = 5, n_estimators = 200)

model.fit(x_train,y_train)



model.score(x_train,y_train)
#Output



preds = model.predict(df_test)

np.exp(preds)



preds = pd.Series(np.exp(preds))



submit = pd.concat([df_test.id, preds], axis = 1)



submit.columns = ['test_id', 'price']



submit.to_csv('submit_result.csv', index = False)