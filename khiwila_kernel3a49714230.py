# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

print('items shape:',items.shape)

print('sample_submission shape:',sample_submission.shape)

print('test shape:',test.shape)

print('train shape:',train.shape)

print('item categories shape:',item_categories.shape)

print('shops shape:',shops.shape)
test_pred = test.copy()

prueba = train.copy()
items.head()
sample_submission.head()
item_categories.head()
items['items_p_cat']= items.groupby(['item_category_id'])['item_id'].transform('count')
shops.head()
train.head()
test.head()
###### MANAGE DATABASES ##########
prueba = prueba.drop(['date','date_block_num','shop_id','item_cnt_day'], axis=1)

prueba = prueba.drop_duplicates(subset ="item_id",  keep='first') 

test = pd.merge(test, prueba, on='item_id', how='left')
def fechas(df):

    dat = df["date"].str.split(".", n = 3, expand = True) 

    df['day']= dat[0]

    df['month']= dat[1].astype('int64')

    df['year']= dat[2].astype('int64')-2012

    df1 = df.drop(['date'], axis=1)

    return df1
test['date_block_num'] =34

test = test[['ID','date_block_num','shop_id','item_id','item_price']]

test['date']="30.11.2015"

test= fechas(test)

train= fechas(train)
test.head()
train = pd.merge(train, items, on='item_id')

test = pd.merge(test, items, on='item_id')
train.head()
def features_train(train):

    train["item_name"] = train["item_name"].astype('category')

    train["Item_name_cod"] = train["item_name"].cat.codes

    

    train['Item_date_count'] = train.groupby(['item_id','date_block_num'])['shop_id'].transform('count')

    train['Shop_date_count'] = train.groupby(['shop_id','date_block_num'])['item_id'].transform('count')

    train['Item_item_count'] = train.groupby(['item_id','date_block_num'])['item_id'].transform('count')

    train['Shop_shop_count'] = train.groupby(['shop_id','date_block_num'])['shop_id'].transform('count')

      

    train['Item_month'] = train.groupby(['month'])['item_id'].transform('sum')

    train['block_num'] = train.groupby(['date_block_num'])['item_id'].transform('sum')

    train['Item_year'] = train.groupby(['year','item_id'])['item_id'].transform('sum')

    train['Item_block'] = train.groupby(['item_id'])['date_block_num'].transform('sum')

        

    train['Item_date_sum'] = train.groupby(['item_id','date_block_num'])['item_id'].transform('sum')

    train['Item_date_max'] = train.groupby(['item_id','date_block_num'])['item_id'].transform('max')

    train['Item_date_min'] = train.groupby(['item_id','date_block_num'])['item_id'].transform('min')

    

    #train['Cat_date_mean'] = train.groupby(['item_category_id','date_block_num'])['item_id'].transform('mean')

    train['Cat_date_max'] = train.groupby(['item_category_id','date_block_num'])['item_id'].transform('max')

    train['Cat_date_min'] = train.groupby(['item_category_id','date_block_num'])['item_id'].transform('min')

    

    train['Price_p_cat_mean'] = train.groupby(['item_category_id'])['item_price'].transform('mean')

    train['Price_p_cat_max'] = train.groupby(['item_category_id'])['item_price'].transform('max')

    train['Price_p_cat_min'] = train.groupby(['item_category_id'])['item_price'].transform('min')

    #train['Price_p_cat_std'] = train.groupby(['item_category_id'])['item_price'].transform('std')

    

    train['shop_by_item_by_year'] = train.groupby(['shop_id','year'])['item_id'].transform('count')

    train['Year_shop_cat'] = train.groupby(['year','shop_id','item_category_id'])['item_id'].transform('count')

    train['Month_shop_cat'] = train.groupby(['month','shop_id','item_category_id'])['item_id'].transform('count')

    train['Block_shop_cat'] = train.groupby(['date_block_num','shop_id','item_category_id'])['item_id'].transform('count')

    

    train['pasta'] = train['item_cnt_day'] * train['item_price']

    

    train['number_one'] = train.groupby(['shop_id','date_block_num'])['item_cnt_day'].transform('sum')

    train['number_two'] = train.groupby(['shop_id','item_id'])['item_cnt_day'].transform('sum')

    #train['number_three'] = train.groupby(['item_id','date_block_num'])['item_cnt_day'].transform('sum')

    train['number_three'] = train.groupby(['shop_id','month'])['item_cnt_day'].transform('sum')

    train['number_four'] = train.groupby(['shop_id','item_id'])['pasta'].transform('sum')

    train['number_five'] = train.groupby(['shop_id','item_id','date_block_num'])['pasta'].transform('sum')

    #train['number_five'] = train.groupby(['item_id','year'])['item_cnt_day'].transform('sum') 

    

    train = train.drop(['item_name','day','pasta'], axis=1)

    return train
train = features_train(train) 
train =train.drop(train[(train.item_cnt_day>=100)].index) # limitacion  item_cnt_day >=20

print(train.shape)

train.head()
test["item_name"] = test["item_name"].astype('category')

test["Item_name_cod"] = test["item_name"].cat.codes



test['Item_date_count'] = test.groupby(['item_id','date_block_num'])['shop_id'].transform('count')

test['Shop_date_count'] = test.groupby(['shop_id','date_block_num'])['item_id'].transform('count')

test['Item_item_count'] = test.groupby(['item_id','date_block_num'])['item_id'].transform('count')

test['Shop_shop_count'] = test.groupby(['shop_id','date_block_num'])['shop_id'].transform('count')



test['Item_month'] = test.groupby(['month'])['item_id'].transform('sum')

test['block_num'] = test.groupby(['date_block_num'])['item_id'].transform('sum')

test['Item_year'] = test.groupby(['year','item_id'])['item_id'].transform('sum')

test['Item_block'] = test.groupby(['item_id'])['date_block_num'].transform('sum')



test['Item_date_sum'] = test.groupby(['item_id','date_block_num'])['item_id'].transform('sum')

test['Item_date_max'] = test.groupby(['item_id','date_block_num'])['item_id'].transform('max')

test['Item_date_min'] = test.groupby(['item_id','date_block_num'])['item_id'].transform('min')



#test['Cat_date_mean'] = test.groupby(['item_category_id','date_block_num'])['item_id'].transform('mean')

test['Cat_date_max'] = test.groupby(['item_category_id','date_block_num'])['item_id'].transform('max')

test['Cat_date_min'] = test.groupby(['item_category_id','date_block_num'])['item_id'].transform('min')



test['uno'] = test.groupby(['item_category_id'])['item_price'].transform('mean')



test['item_price'].fillna(test['uno'], inplace=True)



#test['item_price'].fillna(test['item_price'].mean(), inplace=True)

test['Price_p_cat_mean'] = test.groupby(['item_category_id'])['item_price'].transform('mean')

test['Price_p_cat_max'] = test.groupby(['item_category_id'])['item_price'].transform('max')

test['Price_p_cat_min'] = test.groupby(['item_category_id'])['item_price'].transform('min')

#test['Price_p_cat_std'] = test.groupby(['item_category_id'])['item_price'].transform('std')



test['shop_by_item_by_year'] = test.groupby(['shop_id','year'])['item_id'].transform('count')

test['Year_shop_cat'] = test.groupby(['year','shop_id','item_category_id'])['item_id'].transform('count')

test['Month_shop_cat'] = test.groupby(['month','shop_id','item_category_id'])['item_id'].transform('count')

test['Block_shop_cat'] = test.groupby(['date_block_num','shop_id','item_category_id'])['item_id'].transform('count')



test.sort_values("ID", inplace=True)

test = test.drop(['item_name','ID','day','uno'], axis=1)
test.head()
## DEEP NEURAL NETWORK 



from keras.models import Sequential

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate

from keras.models import Model

from keras.layers.normalization import BatchNormalization 

from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from keras.layers.merge import Concatenate

from keras.layers.core import Lambda, Flatten, Dense, Dropout

from keras.initializers import glorot_uniform

from keras.engine.topology import Layer

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn import metrics

from keras import regularizers

from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
def database_type(db_type):

    

    item_cnt_day= db_type[['item_cnt_day']].copy()

    number_one = db_type[['number_one']].copy()

    number_two = db_type[['number_two']].copy()

    number_three = db_type[['number_three']].copy()

    number_four = db_type[['number_four']].copy()

    number_five = db_type[['number_five']].copy()



    db_type = db_type.drop(['item_cnt_day', 'number_one','number_two','number_three','number_four','number_five'], axis=1)

    #db_type = reduce_mem_usage(db_type)

    

    return [db_type,item_cnt_day, number_one, number_two, number_three, number_four, number_five]
train_csv, val_csv = train_test_split(train, test_size = 0.1, random_state=42)

train_csv = database_type(train_csv)

val_csv = database_type(val_csv)
train_csv[0].head()
test.head()
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

filepath = "weights1.best.hdf5"

best_param =  ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', period=1)



train = {'item_cnt_day': train_csv[1].values,         

        'number_one': train_csv[2].values,

        'number_two': train_csv[3].values,

        'number_three': train_csv[4].values,

        'number_four': train_csv[5].values,

        'number_five': train_csv[6].values}

validation = {'item_cnt_day': val_csv[1].values,         

        'number_one': val_csv[2].values,

        'number_two': val_csv[3].values,

        'number_three': val_csv[4].values,

        'number_four': val_csv[5].values,

        'number_five': val_csv[6].values}



def model_sales(X):

    X_input = Input(shape = (X.shape[1],))

    

    x0 = BatchNormalization()(X_input)



    x1 = Dense(128, activation='relu')(x0)

    x1 = Dense(64, activation='relu')(x1)

    x1 = Dense(32, activation='relu')(x1)

    x1 = Dense(16, activation='relu')(x1)

    x1 = Dense(8, activation='relu')(x1)

    output1  = Dense(1, activation='linear', name = 'number_one')(x1)

     

    x2 = concatenate([x0, output1])

    x2 = Dense(128, activation='relu')(x2)

    x2 = Dense(64, activation='relu')(x2)

    x2 = Dense(32, activation='relu')(x2)

    x2 = Dense(16, activation='relu')(x2)

    x2 = Dense(8, activation='relu')(x2)

    output2  = Dense(1, activation='linear', name = 'number_two')(x2)

    

    x3 = concatenate([x0, output1, output2])

    x3 = Dense(128, activation='relu')(x3)

    x3 = Dense(64, activation='relu')(x3)

    x3 = Dense(32, activation='relu')(x3)

    x3 = Dense(16, activation='relu')(x3)

    x3 = Dense(8, activation='relu')(x3)

    output3  = Dense(1, activation='linear', name = 'number_three')(x3)

    

    x4 = concatenate([x0, output1, output2, output3])

    x4 = Dense(128, activation='relu')(x4)

    x4 = Dense(64, activation='relu')(x4)

    x4 = Dense(32, activation='relu')(x4)

    x4 = Dense(16, activation='relu')(x4)

    x4 = Dense(8, activation='relu')(x4)

    output4  = Dense(1, activation='linear', name = 'number_four')(x4)

    

    x5 = concatenate([x0, output1, output2, output3, output4])

    x5 = Dense(128, activation='relu')(x5)

    x5 = Dense(64, activation='relu')(x5)

    x5 = Dense(32, activation='relu')(x5)

    x5 = Dense(16, activation='relu')(x5)

    x5 = Dense(8, activation='relu')(x5)

    output5  = Dense(1, activation='linear', name = 'number_five')(x5)

    

    x6 = concatenate([x0, output1, output2, output3, output4, output5])

    x6 = Dense(128, activation='relu')(x6)

    x6 = Dense(64, activation='relu')(x6)

    x6 = Dense(32, activation='relu')(x6)

    x6 = Dense(16, activation='relu')(x6)

    x6 = Dense(8, activation='relu')(x6)

    layerday  = Dense(1, activation='linear', name = 'item_cnt_day')(x6)



    model= Model(inputs = [X_input] , outputs = [layerday])



    model.summary()

    return model

    

Model_sales = model_sales(train_csv[0])

Model_sales.compile(loss='mean_squared_error', optimizer='Adam')

history = Model_sales.fit(train_csv[0],train, validation_data=(val_csv[0],validation),epochs=30,verbose=1,batch_size = 256, callbacks=[best_param])
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

_= plt.legend(['Train','Validation'], loc='upper right')
# PREDICTION



Model_sales.load_weights("weights1.best.hdf5")

Model_sales.compile(loss='mean_squared_error', optimizer='Adam')

model_predict = Model_sales.predict(test)
# SUBMISSION:



test_pred['item_cnt_day'] = np.around(model_predict, decimals=0)

test_pred['item_cnt_month'] = test_pred.groupby(['item_id','shop_id'])['item_cnt_day'].transform('sum')

submission = sample_submission.copy()

submission['item_cnt_month'] = test_pred['item_cnt_month'].clip(0,20)

submission.to_csv('submission_SALES.csv', index=False)
test_pred.head(10)
submission.head(10)
# CHECK. VALIDATION SET

val_prueba = val_csv[0].copy()





model_predict_val = Model_sales.predict(val_prueba)

val_prueba['pred_item_cnt_day']= np.around(model_predict_val, decimals=0)

val_prueba['pred_item_cnt_month']= val_prueba.groupby(['item_id','shop_id','date_block_num'])['pred_item_cnt_day'].transform('sum')

val_prueba['item_cnt_day']=val_csv[1]

val_prueba['item_cnt_month'] = val_prueba.groupby(['item_id','shop_id','date_block_num'])['item_cnt_day'].transform('sum').clip(0,20)
val_prueba.head(20)
import math

val_prueba['diff'] = val_prueba['pred_item_cnt_month']- val_prueba['item_cnt_month']

val_prueba['diff2'] = val_prueba['diff']**2

suma = val_prueba.diff2.sum()

t = len(val_prueba)

Rmse = math.sqrt(suma/t)

print ('RMSE =', Rmse)
class ComplexNumber:

    def __init__(self,r ,i ):

        self.real = r

        self.imag = i



    def getData(self):

        print("{0}+{1}j".format(self.real,self.imag))

    



# Create a new ComplexNumber object

c1 = ComplexNumber(2,3)



# Call getData() function

# Output: 2+3j

c1.getData()
c2 = ComplexNumber(1,3)
c2.getData()
del c2
c1.info