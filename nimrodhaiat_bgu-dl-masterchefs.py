# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold,StratifiedKFold
from keras.models import Model
from keras.layers import Input, Embedding, concatenate, Flatten, BatchNormalization, Dense, Dropout, Reshape
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import *
#load data
folder='/kaggle/input/competitive-data-science-predict-future-sales/' 
sales_train_file=pd.read_csv(folder+'sales_train.csv',engine="python")
sales_test_file=pd.read_csv(folder+'test.csv',engine="python")

#fix data
sales_train_file.loc[sales_train_file["shop_id"]== 0, "shop_id"] = 57
sales_train_file.loc[sales_train_file["shop_id"]== 1, "shop_id"] = 58
sales_train_file.loc[sales_train_file["shop_id"]== 23, "shop_id"] = 24
sales_train_file.loc[sales_train_file["shop_id"]== 11, "shop_id"] = 10
sales_train_file
X=sales_train_file[['shop_id','item_id','date_block_num']] 
y=sales_train_file[['item_cnt_day']]
regressor = DecisionTreeRegressor(min_samples_split=50)
score=cross_val_score(regressor, X, y, cv=5,scoring='neg_mean_squared_error')
print("average RMSE:",str((-1*np.average(score))**0.5))
##helper functions for later##
def set_callbacks(description='run1',patience=10):
    es = EarlyStopping(patience=patience,monitor='val_loss')
    rlop = ReduceLROnPlateau(patience=3)   
    cb = [es,rlop] 
    return cb
  
def plot_history(hist):
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()
X=sales_train_file[['shop_id','item_id','item_price']]
y=sales_train_file[['item_cnt_day']]

#look at item_cnt_day distribution
item_count_values=pd.value_counts(y['item_cnt_day']) #count values
item_count_values=item_count_values.head(100) #keep the most popular values
item_count_values.sort_index(inplace=True)
item_count_values.plot.bar(figsize=(20,7)).set_yscale('log')

#decided to limit values to:
y['item_cnt_day'].clip(0.,20.,inplace=True)
#create month\year columns
date_col_in_data_time=pd.to_datetime(sales_train_file['date'])
months=date_col_in_data_time.apply(lambda row: row.month)
years=date_col_in_data_time.apply(lambda row: row.year)

#add them
X['month']=months #add month
X=X.join(pd.get_dummies(years)) #add year
#look at item_price distribution
price_count_values=pd.value_counts(X['item_price']) #count values
price_count_values=price_count_values.head(50) #keep values that are more popular
price_count_values.sort_index(inplace=True)
price_count_values.plot.bar(figsize=(30,4))

#normalize to values [0-39]
X['item_price'].clip(0.0,4000.,inplace=True)
X['price_bin']=pd.cut(X['item_price'], bins=20,labels=[x/20.0 for x in range(20)]).astype('int64')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
shop_length=X['shop_id'].max()+1
item_length=X['item_id'].max()+1
month_length=X['month'].max()+1

shop_inp = Input(shape=(1,))
item_inp = Input(shape=(1,))
month_inp = Input(shape=(1,))

shop_emb = Embedding(shop_length,5,input_length=1, embeddings_regularizer=l2(0.01))(shop_inp)
item_emb = Embedding(item_length,50,input_length=1, embeddings_regularizer=l2(0.01))(item_inp)
month_emb = Embedding(month_length,5,input_length=1, embeddings_regularizer=l2(0.01))(month_inp)

x = concatenate([shop_emb,item_emb,month_emb])
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(128,activation='relu')(x)
x = Dense(8,activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(1)(x)
nn_model = Model([shop_inp,item_inp,month_inp],x)

nn_model.compile(loss = 'mse',optimizer=Adam())
nn_model.summary()
history=nn_model.fit([X_train[f] for f in ["shop_id","item_id","month"]],y_train,epochs=40,validation_data=[[X_val[f] for f in ['shop_id','item_id','month']],y_val],batch_size = 4096, callbacks=set_callbacks())
plot_history(history)
X_test=sales_test_file[['shop_id','item_id']]
X_test['month']=11

X_test=[X_test[f] for f in X_test.columns]
predictions=nn_model.predict(X_test)
preds = pd.DataFrame(predictions, columns=['item_cnt_month'])
preds.to_csv('submission.csv',index_label='ID')
shop_length=X['shop_id'].max()+1
item_length=X['item_id'].max()+1
month_length=X['month'].max()+1
price_length=X['price_bin'].max()+1

shop_inp = Input(shape=(1,))
item_inp = Input(shape=(1,))
price_inp = Input(shape=(1,))
month_inp = Input(shape=(1,))
year2013_inp=Input(shape=(1,))
year2014_inp=Input(shape=(1,))
year2015_inp=Input(shape=(1,))

shop_emb = Embedding(shop_length,5,input_length=1, embeddings_regularizer=l2(0.01))(shop_inp)
item_emb = Embedding(item_length,50,input_length=1, embeddings_regularizer=l2(0.01))(item_inp)
month_emb = Embedding(month_length,5,input_length=1, embeddings_regularizer=l2(0.01))(month_inp)

shop_emb = Reshape((5,))(shop_emb)
item_emb = Reshape((50,))(item_emb)
month_emb = Reshape((5,))(month_emb)

x = concatenate([shop_emb,item_emb,price_inp,month_emb,year2013_inp,year2014_inp,year2015_inp])
x = BatchNormalization()(x)
x = Dense(256,activation='relu')(x)
x = Dense(8,activation='relu')(x)
# x = Dropout(0.1)(x)
x = Dense(1)(x)
nn_model = Model([shop_inp,item_inp,price_inp,month_inp,year2013_inp,year2014_inp,year2015_inp],x)
nn_model.compile(loss = 'mse',optimizer=Adam())
history=nn_model.fit([X_train[f] for f in ['shop_id','item_id','price_bin','month',2013,2014,2015]],y_train,epochs=60,
                 validation_data=[[X_val[f] for f in ['shop_id','item_id','price_bin','month',2013,2014,2015]],y_val],batch_size = 4096, callbacks=set_callbacks())
plot_history(history)
#adding the columns that do not exist in the test file but needed
X_test=sales_test_file[['shop_id','item_id']]
average_prices_for_items=X[['item_id','price_bin']].groupby(['item_id']).mean().to_dict()['price_bin'] #find average price
X_test['price_bin']=X_test.apply(lambda row: average_prices_for_items[row.item_id] if row.item_id in average_prices_for_items else 0,axis=1) #get price for item
X_test['month']=11 #add prediction month

#add one-hot encoder year
X_test[2013]=0 
X_test[2014]=0
X_test[2015]=1
# X_test=[X_test[f] for f in ['shop_id','item_id','price_bin','month',2013,2014,2015]]
# predictions=nn_model.predict(X_test)
# preds = pd.DataFrame(predictions, columns=['item_cnt_month'])
# preds.to_csv('submission.csv',index_label='ID')
nn_model.layers.pop()
regressor = DecisionTreeRegressor(min_samples_split=50)
score=cross_val_score(regressor, nn_model.predict([X[f] for f in ['shop_id','item_id','price_bin','month',2013,2014,2015]]), y, cv=5,scoring='neg_mean_squared_error')
score=[(-1*scr)**0.5 for scr in score] #convert to RMSE
print("average RMSE:",str(np.average(score)))
print("the actual results:")
score