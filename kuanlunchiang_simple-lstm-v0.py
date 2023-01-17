import gc,uuid

import pandas as pd

import numpy as np

import pyarrow as pa

import tensorflow as tf

from pyarrow import parquet as pq

from collections import defaultdict

from sklearn.preprocessing import LabelEncoder
CATEGORICAL_DIM = ['item_id','dept_id','cat_id','store_id','state_id']

windowSize = 7

test_range = windowSize + 10
df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

df.index = df.id

df = df.drop('id',axis = 1)
testDate = list(map(lambda x: 'd_'+str(x),list(range(1913-test_range,1914))))

test = df[CATEGORICAL_DIM+testDate]

train = df.drop(testDate,axis = 1)
# To-Do

# 1. Calendar event generator corresponding to certain date

# 2. Sell Price generator corresponding to certain date

# 3. Make tensorflow dataset

# 4. Make multi-step service



class feature_engineering(object):

    

    def __init__ (self,df,dimList,encoder = LabelEncoder,encodeDict = None):

        super().__init__()

        self._df = df

        self._dimList = dimList

        self.arr = df.drop(dimList,axis =1).values

        self.indexList = df.index.tolist()

        self._encoder = encoder

        self.labelDict,self.encodeDict = self._pandas_to_categorical_encode(encodeDict)





    def _rolling_window(self,a, window):

        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)

        strides = a.strides + (a.strides[-1],)

        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



    def _get_time_tensor(self,arr,window_size):

        tmp = self._rolling_window(arr,window_size+1)

        Xtensor = tmp[:,:-1].reshape(-1,window_size,1)

        Ytensor = tmp[:,-1].reshape(-1,1)

        return (Xtensor,Ytensor)



    def _tensor_factory(self,arr,window_size,categoryIx):

        X,Ytensor = self._get_time_tensor(arr,window_size)

        Xtensor = {}

        for i in self.labelDict:

            label = self.labelDict[i][categoryIx]

            Xtensor[i] = self._label_shape_transform(label,Ytensor.shape)

        Xtensor['sells'] = X

        return (Xtensor,Ytensor)



    def np_to_time_tensor_generator(self,windowSize):

        if np.ndim(self.arr) > 1:

            for ix,v in enumerate(self.arr):

                yield self._tensor_factory(v,windowSize,ix)

        else:

            yield self._tensor_factory(self.arr,windowSize,0) 



    def _label_encode(self,arr,encoder):

        if encoder is None:

            encoder = self._encoder()

            enc_arr = encoder.fit_transform(arr)

        else:

            enc_arr = encoder.transform(arr)

        return enc_arr,encoder



    def _pandas_to_categorical_encode(self,encodeDict):

        if encodeDict is None:

            encodeDict = {}

        labelDict = {}

        for i in self._dimList:

            if i in encodeDict:

                enc_arr,encoder = self._label_encode(self._df[i],encodeDict[i])

            else:

                enc_arr,encoder = self._label_encode(self._df[i],None)

            encodeDict[i] = encoder

            labelDict[i] = enc_arr

        return labelDict,encodeDict



    def _label_shape_transform(self,label,shape):

        tmp = np.zeros(shape)

        tmp += label

        return tmp



    def get_encoder_class(self,label):

        return len(self.encodeDict[label].classes_)

    

    def _get_tf_output_type(self):

        dct = {}

        for i in self.encodeDict:

            dct[i] = tf.int16

        dct['sells'] = tf.float32

        return (dct,tf.float32)

    

    def _get_tf_output_shape(self,window_size):

        dct = {}

        for i in self.encodeDict:

            dct[i] = tf.TensorShape([None,1])

        dct['sells'] = tf.TensorShape([None,window_size,1])

        return (dct,tf.TensorShape([None,1]))

    

    def get_tf_dataset(self,window_size):

        return tf.data.Dataset.from_generator(

                    self.np_to_time_tensor_generator,

                    self._get_tf_output_type(),

                    output_shapes = self._get_tf_output_shape(window_size),

                    args = [window_size]

        ), len(list(self.np_to_time_tensor_generator(window_size)))
fe = feature_engineering(train,['item_id','dept_id','cat_id','store_id','state_id'])

dept_id_class_num = fe.get_encoder_class('dept_id')

state_id_class_num = fe.get_encoder_class('state_id')

store_id_class_num = fe.get_encoder_class('store_id')

cat_id_class_num = fe.get_encoder_class('cat_id')



train_univariate,train_step = fe.get_tf_dataset(windowSize)

train_univariate = train_univariate.prefetch(tf.data.experimental.AUTOTUNE).repeat()
ge = feature_engineering(test,['item_id','dept_id','cat_id','store_id','state_id'],encodeDict= fe.encodeDict)



cacheFile = str(uuid.uuid4())

val_univariate,val_step = ge.get_tf_dataset(windowSize)

vals_univariate = val_univariate.prefetch(tf.data.experimental.AUTOTUNE).repeat()
sells_input = tf.keras.layers.Input(shape=(windowSize,1),name = 'sells')

cat_id_input= tf.keras.layers.Input(shape=1,name = 'cat_id')

store_id_input= tf.keras.layers.Input(shape=1,name = 'store_id')

dept_id_input = tf.keras.layers.Input(shape=1,name = 'dept_id')

state_id_input = tf.keras.layers.Input(shape=1,name = 'state_id')





dept_id = tf.keras.layers.Embedding(dept_id_class_num,1)(dept_id_input)

dept_id =tf.keras.layers.Flatten()(dept_id)



state_id = tf.keras.layers.Embedding(dept_id_class_num,1)(state_id_input)

state_id =tf.keras.layers.Flatten()(state_id)



cat_id = tf.keras.layers.Embedding(cat_id_class_num,1)(cat_id_input)

cat_id =tf.keras.layers.Flatten()(cat_id)



store_id = tf.keras.layers.Embedding(store_id_class_num,1)(store_id_input)

store_id =tf.keras.layers.Flatten()(store_id)



lstm = tf.keras.layers.LSTM(3)(sells_input)

lstm = tf.keras.layers.Dense(20,'relu')(lstm)



dense = tf.keras.layers.Concatenate()([lstm,cat_id,store_id,dept_id,state_id])

dense = tf.keras.layers.Dense(40,'relu')(dense)

dense = tf.keras.layers.Dropout(0.2)(dense)

dense = tf.keras.layers.Dense(15,'relu')(dense)

dense = tf.keras.layers.Dense(1,'relu')(dense)

simple_lstm_model = tf.keras.models.Model({

    'sells':sells_input,

    'cat_id':cat_id_input,

    'dept_id':dept_id_input,

    'state_id':state_id_input,

    'store_id':store_id_input

},dense)

simple_lstm_model.compile(optimizer='adam', loss='mse',metrics = ['mae','mse'])
tf.keras.utils.plot_model(simple_lstm_model,show_shapes=True)
EVALUATION_INTERVAL = len(list(fe.np_to_time_tensor_generator(windowSize)))

validation_steps = len(list(ge.np_to_time_tensor_generator(windowSize)))

EPOCHS = 10
simple_lstm_model.fit(

    train_univariate,

    epochs=EPOCHS,

    steps_per_epoch= EVALUATION_INTERVAL,

    validation_data=vals_univariate, 

    validation_steps=validation_steps

    )
gen = ge.np_to_time_tensor_generator(windowSize)

for ix,v in enumerate(gen):

    if ix == 0:

        dct = v[0]

        continue

    for i in v[0]:

        dct[i] = np.concatenate((dct[i],v[0][i]))

prd = simple_lstm_model.predict(dct, verbose =1)

prd = prd.reshape(-1,11)
for i in range(20):

    prdDf = pd.DataFrame(prd[i].reshape((1,-1)),columns = testDate[-11:])

    prdDf.append(test[prdDf.columns].head(i+1).tail(1)).transpose().plot()
prd
prdDate = list(map(lambda x: 'd_'+str(x),list(range(1914-windowSize,1914))))

prdDf = df[CATEGORICAL_DIM+prdDate]
for i in range(28):

    print(i)

    dtList = prdDf.columns.tolist()

    [dtList.remove(i) for i in CATEGORICAL_DIM]

    dtList = dtList[-windowSize:]

    prdDf = prdDf[CATEGORICAL_DIM+dtList]



    prdDf['d_'+ str(int(prdDf.columns.tolist()[-1].split('_')[-1])+1)] = -1

    pe = feature_engineering(prdDf,['item_id','dept_id','cat_id','store_id','state_id'],encodeDict= fe.encodeDict)

    gen = pe.np_to_time_tensor_generator(windowSize)

    prd = simple_lstm_model.predict_generator(gen,use_multiprocessing = True, verbose =1)

    prdDf[prdDf.columns[-1]] = list(map(lambda x: float(x[0]) ,prd))