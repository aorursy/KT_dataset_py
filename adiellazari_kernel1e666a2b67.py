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
import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

%matplotlib inline

data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

item_cats = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
print ('Data Set') 

print(data.shape)



print ('Test Set') 

print(test.shape)

data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')

data['month'] = data['date'].dt.month

data['year'] = data['date'].dt.year



data = data.drop(['date', 'item_price'], axis=1)

data = data.groupby([c for c in data.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()

data = data.rename(columns={'item_cnt_day':'item_cnt_month'})



data.head()
shop_item_mean = data[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()

shop_item_mean = shop_item_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})
shop_prev_month = data[data['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]

shop_prev_month = shop_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_mean'})
data = pd.merge(data, shop_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.)

data = pd.merge(data, items, how='left', on='item_id')

data = pd.merge(data, item_cats, how='left', on='item_category_id')

data = pd.merge(data, shops, how='left', on='shop_id')

data.head()
test['month'] = 11

test['year'] = 2015

test['date_block_num']=34

test = pd.merge(test, shop_item_mean, how='left', on=['shop_id', 'item_id']).fillna(0.)

test = pd.merge(test, shop_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.)

test = pd.merge(test, items, how='left', on='item_id')

test = pd.merge(test, item_cats, how='left', on='item_category_id')

test = pd.merge(test, shops, how='left', on='shop_id')

test['item_cnt_month'] = 0.

test.head()
from sklearn import preprocessing

for c in ['shop_name', 'item_name', 'item_category_name']:

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(data[c].unique())+list(test[c].unique()))

    data[c] = lbl.transform(data[c].astype(str))

    test[c] = lbl.transform(test[c].astype(str))
col = [c for c in data.columns if c not in ['item_cnt_month']]

x1 = data[data['date_block_num']<33]

y1 = x1['item_cnt_month']

x1=x1[col]

x2 = data[data['date_block_num'] == 33]

y2 = x2['item_cnt_month'].to_numpy()

x2=x2[col]
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0).fit(x1,y1) 

# from sklearn.svm import SVC

# clf = SVC(gamma='auto').fit(x1,y1) 

pred = clf.predict(x2)
from sklearn.metrics import accuracy_score

accuracy_score(y2, pred)



test['item_cnt_month'] = clf.predict(test[col])

test[['ID', 'item_cnt_month']].to_csv('../working/submission.csv', index=False)
# from tensorflow.keras.layers import Concatenate, Embedding, Input, Reshape

from keras.models import Model

from keras.layers import *

from keras.callbacks import *

from keras.regularizers import l2

from keras.optimizers import *

from keras.utils import to_categorical
cat_features = ['date_block_num', 'item_id', 'shop_id', 'item_category_id','month', 'year']#,'month_length'

lenUniquFeatures = {}

processed_data = data.loc[:,cat_features].copy()

for cat_feature in cat_features:

    feature_inx ={p:i for (i,p) in enumerate(data[cat_feature].unique())}

    lenUniquFeatures[cat_feature] = len(feature_inx)

    processed_data[cat_feature] = [feature_inx[x] for x in data[cat_feature]]

target = data['item_cnt_month'].values

#cat_features_train_test = pd.concat([train[cat_features], test[cat_features]])



processed_data
def build_categorical_inputs(features):

    initial_inputs = {}

    cat_input_layers={}

    for feature in features:

#         print(cat_features[feature])

#         num_unique_cat = cat_features[feature].nunique()

             

        embedding_size = int(min(np.sqrt(lenUniquFeatures[feature]), 50))

#         categories  = num_unique_cat + 1



        initial_inputs[feature] = Input(shape=(1,))

        embedding_layer = Embedding(lenUniquFeatures[feature], 

                                    embedding_size,

                                    embeddings_regularizer=l2(0.001),

                                    input_length=1)(initial_inputs[feature])

        cat_input_layers[feature] = Reshape(target_shape=(embedding_size,))(embedding_layer)



    return initial_inputs, cat_input_layers

initial_inputs, input_layers = build_categorical_inputs(cat_features)
x = Concatenate(axis=-1)([layer for layer in input_layers.values()])

x = BatchNormalization()(x)

x = Dense(128,activation='relu')(x)

x = Dense(64,activation='relu')(x)

x = Dropout(0.2)(x)

x = BatchNormalization()(x)

x = Dense(32,activation='relu')(x)

x = Dense(16,activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(2,activation='softmax')(x)

model = Model([input for input in initial_inputs.values()],x)

model.compile(loss = 'categorical_crossentropy',optimizer='RMSProp')

model.summary()
from sklearn.model_selection import KFold,StratifiedKFold



def set_callbacks(description='run1',patience=15,tb_base_logdir='./logs/'):

    cp = ModelCheckpoint('best_model_weights_{}.h5'.format(description),save_best_only=True)

    es = EarlyStopping(patience=patience,monitor='val_loss')

    rlop = ReduceLROnPlateau(patience=5)   

    tb = TensorBoard(log_dir='{}{}'.format(tb_base_logdir,description))

#     cycl = CyclicLR(max_lr=0.03,step_size=5000)

    cb = [cp,es,tb,rlop]

    return cb



features = processed_data.columns

kf = StratifiedKFold(n_splits=5,random_state=42)

# kf = KFold(n_splits=5,random_state=42)

for tr_ind,val_ind in kf.split(processed_data,target):

    X_train,X_val,y_train,y_val = processed_data.iloc[tr_ind],processed_data.iloc[val_ind],target[tr_ind],target[val_ind]

    model.fit([X_train[f] for f in features],to_categorical(y_train),epochs=25,

                 validation_data=[[X_val[f] for f in features],to_categorical(y_val)],callbacks=set_callbacks())

    break