

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

import folium 



import os

import gc

import joblib



import time



from sklearn import metrics, linear_model

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn.preprocessing import StandardScaler

from tqdm.notebook import tqdm







import tensorflow as tf

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics, preprocessing

from tensorflow.keras import layers

from tensorflow.keras import optimizers

from tensorflow.keras.models import Model, load_model

from tensorflow.keras import callbacks

from tensorflow.keras import backend as K

from tensorflow.keras import utils



from sklearn import model_selection

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error

from math import sqrt



import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('../input/chicago-traffic/chicago.csv')

data["TIME"]=pd.to_datetime(data["TIME"], format="%m/%d/%Y %H:%M:%S %p")

data.drop(data[data['SPEED']==0].index,inplace =True)

data['day'] = data['TIME'].dt.day

data['MONTH'] = data['TIME'].dt.month

data['YEAR'] = data['TIME'].dt.year

data = data[data['YEAR']!=2020]

data = data[data['SPEED']<100]



data = data.groupby(['REGION_ID','HOUR','MONTH','day', 'WEST','EAST', 'SOUTH','NORTH','DAY_OF_WEEK','YEAR'])[['SPEED','BUS_COUNT','NUM_READS']].agg('mean').reset_index()

data['CENTER_LAT']=data['NORTH']*0.5+0.5*data['SOUTH']

data['CENTER_LON']=data['EAST']*0.5+0.5*data['WEST']
data['MINUTE'] = '00'

data['Time'] = pd.to_datetime(data[['YEAR','MONTH','day','HOUR','MINUTE']].astype(str).agg('-'.join,axis=1),format='%Y-%m-%d-%H-%M')
def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h
def width(x) : 

    return haversine_array(x['NORTH'],x['WEST'],x['NORTH'],x['EAST'])

def length(x) : 

    return haversine_array(x['NORTH'],x['EAST'],x['SOUTH'],x['EAST'])
tqdm.pandas()

data['length'] =  data[['WEST','EAST', 'SOUTH','NORTH']].progress_apply(length,axis=1)

data['width']  =  data[['WEST','EAST', 'SOUTH','NORTH']].progress_apply(width,axis=1)
data['area'] = data['length']*data['width']

data['reders_per_area'] = data['BUS_COUNT']/data['area'] 

data['READS_per_area'] = data['NUM_READS']/data['area'] 

data['BUS_ratio'] = data['BUS_COUNT']/data['NUM_READS'] 
categorical_features = ['REGION_ID','MONTH','HOUR','day','DAY_OF_WEEK','YEAR']

Numerical_features = ['NUM_READS','BUS_COUNT','area','length','width','CENTER_LAT','CENTER_LON','reders_per_area','READS_per_area','BUS_ratio']
for categorical_var in categorical_features:

    

    cat_emb_name= categorical_var.replace(" ", "")+'_Embedding'

  

    no_of_unique_cat  = data[categorical_var].nunique()

    embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 64))

  

    print('Categorica Variable:', categorical_var,

        'Unique Categories:', no_of_unique_cat,

        'Embedding Size:', embedding_size)
def create_model(data, categorical_features ,  Numerical_features ):    

    input_models=[]

    output_embeddings=[]

    for categorical_var in categorical_features  :

        cat_emb_name= categorical_var.replace(" ", "")+'_Embedding'

        no_of_unique_cat  = data[categorical_var].nunique() +1

        embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 24 ))

        

        input_model = layers.Input(shape=(1,),name=cat_emb_name)

        output_model = layers.Embedding(no_of_unique_cat, embedding_size, name=cat_emb_name+'emblayer')(input_model)

        output_model = layers.Reshape(target_shape=(embedding_size,))(output_model)    

        input_models.append(input_model)

        output_embeddings.append(output_model)



    for c in Numerical_features :

        num_name= c.replace(" ", "")+'_num'

        input_numeric = layers.Input(shape=(1,),name= num_name)

        embedding_numeric = layers.Dense(16, kernel_initializer="uniform")(input_numeric) 

        input_models.append(input_numeric)

        output_embeddings.append(embedding_numeric)



  



    #At the end we concatenate altogther and add other Dense layers

    output = layers.Concatenate()(output_embeddings)



    output = layers.Dense(1024, kernel_initializer="uniform")(output)

    output = layers.Activation('relu')(output)

    output= layers.Dropout(0.5)(output)

    output = layers.Dense(512, kernel_initializer="uniform")(output)

    output = layers.Activation('relu')(output)

    output= layers.Dropout(0.3)(output)

    output = layers.Dense(256, kernel_initializer="uniform")(output)

    output = layers.Activation('relu')(output)

    output= layers.Dropout(0.1)(output)

    output = layers.Dense(1)(output)



    model = Model(inputs=input_models, outputs=output)

    return model 
model = create_model(data , categorical_features ,Numerical_features)

model.summary()
from sklearn.preprocessing import StandardScaler

for num in Numerical_features :

    scalar=StandardScaler()

    scalar.fit(data[num].values.reshape(-1, 1))

    data[num]=scalar.transform(data[num].values.reshape(-1, 1)) 

    data[num]=scalar.transform(data[num].values.reshape(-1, 1)) 
#converting data to list format to match the network structure

def preproc(X_train, X_val, X_test):



    input_list_train = dict()

    input_list_val = dict()

    input_list_test = dict()

    

    #the cols to be embedded: rescaling to range [0, # values)

    for c in categorical_features :

        cat_emb_name= c.replace(" ", "")+'_Embedding'

        raw_vals = X_train[c].unique()

        val_map = {}

        for i in range(len(raw_vals)):

            val_map[raw_vals[i]] = i       

        input_list_train[cat_emb_name]=X_train[c].map(val_map).values

        input_list_val[cat_emb_name]=X_val[c].map(val_map).fillna(0).values

        input_list_test[cat_emb_name]=X_test[c].map(val_map).fillna(0).values

    for c in Numerical_features :

        num_name= c.replace(" ", "")+'_num'

               

        input_list_train[num_name]=X_train[c].values

        input_list_val[num_name]=X_val[c].values

        input_list_test[num_name]=X_test[c].values



    

    return input_list_train, input_list_val, input_list_test
X_train , X_test  = model_selection.train_test_split(data , test_size = 0.1 , random_state=44 , shuffle =True)

X_train , X_vaild = model_selection.train_test_split(X_train , test_size = 0.2 , random_state=44 , shuffle =True)
y_train,y_valid,y_test = X_train.SPEED,X_vaild.SPEED,X_test.SPEED

X_train,X_vaild,X_test = preproc(X_train,X_vaild,X_test)
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

  # Restrict TensorFlow to only use the first GPU

  try:

    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    logical_gpus = tf.config.experimental.list_logical_devices('GPU')

    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

  except RuntimeError as e:

    # Visible devices must be set before GPUs have been initialized

    print(e)
EPOCHS = 35

BATCH_SIZE =1048

AUTO = tf.data.experimental.AUTOTUNE

train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_train, y_train))

    .repeat() 

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_vaild, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)
start_time = time.time()

model = create_model(data , categorical_features ,Numerical_features)

es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,

                                 verbose=5, baseline=None, restore_best_weights=True)

rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,

                                      patience=3, min_lr=1e-6, mode='max', verbose=1)

model.compile(optimizer = Adam(lr=5e-5), loss = 'mean_squared_error', metrics =[tf.keras.metrics.RootMeanSquaredError()])

n_steps = sum( [x.shape[0] for x in X_train.values()] ) // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS

)





              
print("total training time is %s Minute " %((time.time() - start_time)/60))

print("hardware : NVidia K80 GPUs ")


# summarize history for accuracy

plt.plot(train_history.history['loss'])

plt.plot(train_history.history['val_loss'])

plt.title('loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper right')

plt.show()

!mkdir infrastructure_project

model.save_weights('infrastructure_project/Model_1.ckpt')
def rmse(predictions, targets): 

    return sqrt(mean_squared_error(predictions, targets))
valid_fold_preds = model.predict(X_test)

print('TEST RMSE = :' , rmse(y_test.values, valid_fold_preds  ))
def plot(region , year , month , day ) : 

    sub_plot = data[(data['REGION_ID']==region)&(data['YEAR']==year)&(data['MONTH']==month)& (data['day']<day)]

    sub_plot = sub_plot.sort_values('Time')

    y=  sub_plot.SPEED

    X,_,_ = preproc(sub_plot,sub_plot,sub_plot)

    predictions  = model.predict(X)

    plt.figure(figsize=(24, 8))

    plt.plot(sub_plot['Time'].values , sub_plot['SPEED'].values, '--',label = 'real values')

    plt.plot(sub_plot['Time'].values , predictions,label = 'predicted values')

    plt.ylabel(f'SPEED in region id {region}')

    plt.title(f' predicted vs real values ')

    plt.xlabel('Time')

    plt.grid(True)

    plt.legend()



    plt.show()
plot(1,2019,1,10)
plot(5,2018,4,14)