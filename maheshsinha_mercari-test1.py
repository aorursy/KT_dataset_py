import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



train_ds = pd.read_csv('../input/train.tsv', sep='\t')

test_ds = pd.read_csv('../input/test.tsv', sep='\t', quoting=3, error_bad_lines=False)



train_ds.info()



#handel missing values:



def missing_val(dataset):

    dataset['category_name'].fillna('missing', inplace=True)

    dataset['brand_name'].fillna('missing', inplace=True)

    dataset['item_description'].fillna('missing', inplace=True)

    return dataset

    

train_ds = missing_val(train_ds)

test_ds = missing_val(test_ds)



#split the category columns in sub columns:



def split_catg_name(row):

    try:

        text = row

        text1, text2, text3 = text.split('/')

        return text1, text2, text3

    except:

        return 'no label', 'no label', 'no label'

    

train_ds['gen_cat'], train_ds['sub_cat1'], train_ds['sub_cat2'] = zip(*train_ds['category_name'].apply(lambda x : split_catg_name(x)))

test_ds['gen_cat'], test_ds['sub_cat1'], test_ds['sub_cat2'] = zip(*test_ds['category_name'].apply(lambda x : split_catg_name(x)))



#drop category_name

train_ds.drop('category_name', axis=1, inplace=True)

test_ds.drop('category_name', axis=1, inplace=True)



#use label encoder on categorical data:

from sklearn.preprocessing import LabelEncoder



def lab_encoder(dataset):

    le = LabelEncoder()

    dataset['gen_cat'] = le.fit_transform(dataset['gen_cat'])

    dataset['sub_cat1'] = le.fit_transform(dataset['sub_cat1'])

    dataset['sub_cat2'] = le.fit_transform(dataset['sub_cat2'])

    dataset['brand_name'] = le.fit_transform(dataset['brand_name'])

    dataset['name'] = le.fit_transform(dataset['name'])

    return dataset



train_ds = lab_encoder(train_ds)

test_ds = lab_encoder(test_ds)



#drop item_description from test and train datasets:



train_ds.drop('item_description', axis=1, inplace=True)

test_ds.drop('item_description', axis=1, inplace=True)



#split datasets in train and test sets:

from sklearn.cross_validation import train_test_split

train_X, test_X, train_y, test_y = train_test_split(train_ds.drop('price', axis=1), train_ds['price'], test_size = 0.2)



#apply Random Forest regression algorithm:

from sklearn.ensemble import RandomForestRegressor

rand_reg2 = RandomForestRegressor(n_estimators=10)

rand_reg2.fit(train_X, train_y)

rand_reg2.score(train_X, train_y)



y_pred_RFR = rand_reg2.predict(test_ds)

prediction = pd.DataFrame({'test_id': test_ds['test_id'],

                          'price': y_pred_RFR})





#sub = pd.read_csv('../input/sample_submission.csv')

prediction.to_csv('First_Kernel.csv', index=False)