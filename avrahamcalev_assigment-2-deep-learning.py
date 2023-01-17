import tensorflow as tf
import pandas as pd
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
submit = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
print('Number of shops: ', len(shops))
print('Number of item categories: ',len(categories))
print('Number of items: ', len(items))
print('size of the data: ',data.shape)
print('0:[date] 1:[date_block_num] 2:[shop_id] 3:[item_id] 4:[item_price] 5:[item_cnt_day]')
print('Total sold items from all month ', sum(data['item_cnt_day']))
print('submit: number of items to predict there sold amount ', len(submit))
import matplotlib.pyplot as plt
pre_process_data = data.drop(['date','item_price','shop_id','item_id'], axis=1)
sum_m = pre_process_data.groupby(['date_block_num']).sum()
sum_m.plot()
pre_process_data = data.drop('date', axis=1)
avg_prices = pre_process_data.groupby(['date_block_num','shop_id','item_id']).mean().reset_index()
X = avg_prices.drop('item_cnt_day', axis=1)
X = X.rename(index=str, columns = {"item_price":"price_month", 'date_block_num':'month_num'})
cat_items = items.drop('item_name', axis=1)
X = X.merge(cat_items, on='item_id')
X
pre_process_data = data.drop(['date', 'item_price'], axis=1)
sales = pre_process_data.groupby(['date_block_num','shop_id','item_id']).sum().reset_index()
y = sales.rename(index=str, columns = {'date_block_num':'month_num'})
y
#create embedding for month
months = {p:i for (i,p) in enumerate(X['month_num'].unique())}
#create embedding for store
stores = {p:i for (i,p) in enumerate(X['shop_id'].unique())}
#create embedding for category
cates = {p:i for (i,p) in enumerate(X['item_category_id'].unique())}
#create embedding for product
products = {p:i for (i,p) in enumerate(X['item_id'].unique())}
#create pre process data from the embedings
preprocess = X.loc[:,['month_num','shop_id','item_id','item_category_id']].copy()
preprocess['month_num'] = [months[x] for x in X['month_num']]
preprocess['shop_id'] = [stores[x] for x in X['shop_id']]
preprocess['item_category_id'] = [cates[x] for x in X['item_category_id']]
preprocess['item_id'] = [products[x] for x in X['item_id']]
#taking last month for testing our model
X_train = preprocess[preprocess['month_num']!=33]
y_train = y[y['month_num']!=33]['item_cnt_day']
X_test = preprocess[preprocess['month_num']==33]
y_test = y[y['month_num']==33]['item_cnt_day']
print('train: X ', X_train.shape, ' y', y_train.shape)
print('test: X ', X_test.shape, ' y', y_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
reg = LinearRegression()
reg.fit(X_train,y_train)
preds = reg.predict(X_test)
print('Linear regression loss:',mean_squared_error(y_test,preds))
from keras.models import Model
from keras.layers import Input, Embedding, concatenate, Dense, Flatten
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import numpy as np
input_store = Input(shape=(1,), dtype='int64')
input_product =Input(shape=(1,), dtype='int64')

embeding_store = Embedding(len(stores), 5, input_length=1, embeddings_regularizer=l2(1e-6) )(input_store)
embeding_product = Embedding(len(products), 5, input_length=1, embeddings_regularizer=l2(1e-6) )(input_product)
x = concatenate([embeding_store, embeding_product])
x = Flatten()(x)
x = Dense(1, activation='relu')(x)

model = Model([input_store,input_product],x)
model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')
model.summary()
early_stopping_monitor = EarlyStopping(patience=3)

history = model.fit([X_train['shop_id'],X_train['item_id']], 
                    y_train, validation_split=0.2, callbacks=[early_stopping_monitor], shuffle=True, epochs=10)
import matplotlib.pyplot as plt

def show_graphs(history):
  # Plot training & validation accuracy values
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show()
show_graphs(history)
from sklearn.metrics import confusion_matrix,accuracy_score, mean_squared_error, mean_absolute_error
preds = model.predict([X_test['shop_id'],X_test['item_category_id']])
print("Test mean square error: ", mean_squared_error(y_test, preds))
print("Test mean_absolute_error: ", mean_absolute_error(y_test, preds))
cat_items = items.drop('item_name', axis=1)
submit = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
submit = submit.merge(cat_items, on=['item_id'])
submit['month_num'] = '34' # it is the next value of month 34 => october 2015

#embeddings
stores = {p:i for (i,p) in enumerate(submit['shop_id'].unique())}
cates = {p:i for (i,p) in enumerate(submit['item_category_id'].unique())}
products = {p:i for (i,p) in enumerate(submit['item_id'].unique())}

#create pre process data from the embedings
submit_preprocess = submit.loc[:,['month_num','shop_id','item_id','item_category_id']].copy()
submit_preprocess['shop_id'] = [stores[x] for x in submit['shop_id']]
submit_preprocess['item_category_id'] = [cates[x] for x in submit['item_category_id']]
submit_preprocess['item_id'] = [products[x] for x in submit['item_id']]
preds = model.predict([submit_preprocess['shop_id'], submit_preprocess['item_category_id']])
output = pd.DataFrame({'ID':range(0,len(preds))})
output['item_cnt_month'] = pd.Series(np.max(preds,axis=-1), index=output.index)
output.to_csv('outD.csv',index=False)
from keras.models import Model
from keras.layers import Input, Embedding, concatenate, Dense, Flatten, Dropout, LSTM, RepeatVector 
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import numpy as np

months = {p:i for (i,p) in enumerate(X['month_num'].unique())}
stores = {p:i for (i,p) in enumerate(X['shop_id'].unique())}
cates = {p:i for (i,p) in enumerate(X['item_category_id'].unique())}
products = {p:i for (i,p) in enumerate(X['item_id'].unique())}

input_month = Input(shape=(1,), dtype='int64')
input_store = Input(shape=(1,), dtype='int64')
input_cat = Input(shape=(1,), dtype='int64')
input_product =Input(shape=(1,), dtype='int64')

embeding_month = Embedding(len(months)+1, 5, input_length=1, embeddings_regularizer=l2(1e-6))(input_month)
embeding_store = Embedding(len(stores), 5, input_length=1, embeddings_regularizer=l2(1e-6) )(input_store)
embeding_cat = Embedding(len(cates), 5, input_length=1, embeddings_regularizer=l2(1e-6) )(input_cat)
embeding_product = Embedding(len(products), 5, input_length=1, embeddings_regularizer=l2(1e-6) )(input_product)

x = concatenate([embeding_month, embeding_store, embeding_cat, embeding_product])
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(4, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(48, activation='relu')(x)
x = Dense(192, activation='relu')(x)
x = Dense(1, activation='relu')(x)
model = Model([input_month,input_store,input_cat,input_product],x)
model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')
model.summary()
early_stopping_monitor = EarlyStopping(patience=3) 
history = model.fit([X_train['month_num'],X_train['shop_id'],X_train['item_category_id'],X_train['item_id']], 
                    y_train, validation_split=0.2, callbacks=[early_stopping_monitor], shuffle=True, epochs=10)
show_graphs(history)
from sklearn.metrics import confusion_matrix,accuracy_score, mean_squared_error, mean_absolute_error
preds = model.predict([X_test['month_num'],X_test['shop_id'],X_test['item_category_id'],X_test['item_id']])
print("Test mean square error: ", mean_squared_error(y_test, preds))
print("Test mean_absolute_error: ", mean_absolute_error(y_test, preds))
cat_items = items.drop('item_name', axis=1)
submit = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
submit = submit.merge(cat_items, on=['item_id'])
submit['month_num'] = '34' # it is the next value of month 34 => october 2015

#embeddings
stores = {p:i for (i,p) in enumerate(submit['shop_id'].unique())}
cates = {p:i for (i,p) in enumerate(submit['item_category_id'].unique())}
products = {p:i for (i,p) in enumerate(submit['item_id'].unique())}

#create pre process data from the embedings
submit_preprocess = submit.loc[:,['month_num','shop_id','item_id','item_category_id']].copy()
submit_preprocess['shop_id'] = [stores[x] for x in submit['shop_id']]
submit_preprocess['item_category_id'] = [cates[x] for x in submit['item_category_id']]
submit_preprocess['item_id'] = [products[x] for x in submit['item_id']]
preds = model.predict([submit_preprocess['month_num'],submit_preprocess['shop_id'],submit_preprocess['item_category_id'],submit_preprocess['item_id']])
output = pd.DataFrame({'ID':range(0,len(preds))})
output['item_cnt_month'] = pd.Series(np.max(preds,axis=-1), index=output.index)
output.to_csv('outE.csv',index=False)
def similar(vecA, vecB, treshold):
    if len(vecA) != len(vecB):
        return False
    for i in range(0, len(vecA)):
        if abs(float(vecA[i])-float(vecB[i]))> treshold:
            return False
    return True

def get_simalirty_map(metrix, treshold=0.001):
    map = pd.DataFrame({})
    map['vector A'] = pd.Series(range(0,len(metrix)))
    map['similar to vector B'] = ''
    np_data = metrix.to_numpy()
    for i,vecA in enumerate(np_data):
        for j,vecB in enumerate(np_data):
            if i != j and similar(vecA, vecB, treshold):
                value = map.iloc[i, map.columns.get_loc('similar to vector B')]
                if value == '':
                    map.iloc[i, map.columns.get_loc('similar to vector B')] = str(j)
                else:
                    map.iloc[i, map.columns.get_loc('similar to vector B')] = value + ', '+str(j)
    return map

def get_embbeded_matrix(weights):
    db = pd.DataFrame({})
    for i in range(0,len(weights[0])):
        v = weights[:,i]
        if i==0:
            db[str(i)] = pd.Series(v)
        else:
            db[str(i)] = pd.Series(v, index=db.index)
    return db
w =  model.get_weights()
month_weigts = np.array(w[0])
month_metrix = get_embbeded_matrix(month_weigts)
month_simalrty = get_simalirty_map(month_metrix, treshold=0.7)
# this is frame of the similar months
month_simalrty[month_simalrty['similar to vector B']!='']
store_weigts = np.array(w[1])
store_metrix = get_embbeded_matrix(store_weigts)
store_simalrty = get_simalirty_map(store_metrix,treshold=0.7)
store_simalrty[store_simalrty['similar to vector B']!='']
cat_weights = np.array(w[2])
cat_metrix = get_embbeded_matrix(cat_weights)
cat_simalrty = get_simalirty_map(cat_metrix,treshold=0.01)
cat_simalrty[cat_simalrty['similar to vector B']!='']
product_weights = np.array(w[3])
product_metrix = get_embbeded_matrix(product_weights[:100,])
product_simalrty = get_simalirty_map(product_metrix,treshold=0.07)
product_simalrty[product_simalrty['similar to vector B']!='']
from keras.models import Model
from keras.layers import Input, Embedding, concatenate, Dense, Flatten, Dropout, LSTM, RepeatVector 
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import numpy as np

months = {p:i for (i,p) in enumerate(X['month_num'].unique())}
stores = {p:i for (i,p) in enumerate(X['shop_id'].unique())}
cates = {p:i for (i,p) in enumerate(X['item_category_id'].unique())}
products = {p:i for (i,p) in enumerate(X['item_id'].unique())}

input_month = Input(shape=(1,), dtype='int64')
input_store = Input(shape=(1,), dtype='int64')
input_cat = Input(shape=(1,), dtype='int64')
input_product =Input(shape=(1,), dtype='int64')

embeding_month =  Embedding( len(months)+1, 5, input_length=1, embeddings_regularizer=l2(1e-6), weights = model.layers[4].get_weights())(input_month)
embeding_store = Embedding(len(stores), 5, input_length=1, embeddings_regularizer=l2(1e-6) , weights = model.layers[5].get_weights())(input_store)
embeding_cat = Embedding(len(cates), 5, input_length=1, embeddings_regularizer=l2(1e-6) ,weights = model.layers[6].get_weights())(input_cat)
embeding_product = Embedding(len(products), 5, input_length=1, embeddings_regularizer=l2(1e-6) , weights = model.layers[7].get_weights())(input_product)

x = concatenate([embeding_month, embeding_store, embeding_cat, embeding_product],weights = model.layers[8].get_weights())
x = Flatten(weights = model.layers[9].get_weights())(x)

model = Model([input_month,input_store,input_cat,input_product],x)
model.summary()
embedding_train = model.predict([X_train['month_num'],X_train['shop_id'],X_train['item_category_id'],X_train['item_id']])
embedding_test = model.predict([X_test['month_num'],X_test['shop_id'],X_test['item_category_id'],X_test['item_id']])
#reshape the results from predict to train logistic Regression
e_train = np.squeeze(embedding_train)
e_test = np.squeeze(embedding_test)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
reg = LinearRegression()
reg.fit(e_train,y_train)
preds = reg.predict(e_test)
print('Linear regression loss:',mean_squared_error(y_test,preds))
cat_items = items.drop('item_name', axis=1)
submit = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
submit = submit.merge(cat_items, on=['item_id'])
submit['month_num'] = '34' # it is the next value of month 34 => october 2015

#embeddings
stores = {p:i for (i,p) in enumerate(submit['shop_id'].unique())}
cates = {p:i for (i,p) in enumerate(submit['item_category_id'].unique())}
products = {p:i for (i,p) in enumerate(submit['item_id'].unique())}

#create pre process data from the embedings
submit_preprocess = submit.loc[:,['month_num','shop_id','item_id','item_category_id']].copy()
submit_preprocess['shop_id'] = [stores[x] for x in submit['shop_id']]
submit_preprocess['item_category_id'] = [cates[x] for x in submit['item_category_id']]
submit_preprocess['item_id'] = [products[x] for x in submit['item_id']]
preds = model.predict([submit_preprocess['month_num'],submit_preprocess['shop_id'],submit_preprocess['item_category_id'],submit_preprocess['item_id']])
# preds = dt.predict(preds)
preds = reg.predict(preds)

output = pd.DataFrame({'ID':range(0,len(preds))})
output['item_cnt_month'] = pd.Series(np.max(preds,axis=-1), index=output.index)
output.to_csv('outG.csv',index=False)