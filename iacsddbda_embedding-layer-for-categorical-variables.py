import pandas as pd



import tensorflow as tf



import keras



from keras.layers import Dense,Input,BatchNormalization,Dropout,Flatten



from keras.models import Sequential

data=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.isna().sum()

data.dropna()

ii=data.index[data.price==0]

data.drop(ii, inplace=True)# there are 11 records for which price is 0

data.room_type= data.room_type.apply(lambda x:0 if x=='Private room' else 1 if x=='Entire home/apt' else 2 if x=='Shared room' else None)

from sklearn.utils import shuffle

data = shuffle(data)
data.sample(random_state=1)

my_data=data.iloc[:40000,([2,4,5,9,10])]

y=data.iloc[:40000,8]

y=data.iloc[:40000,8]

y=keras.utils.to_categorical(y,3)


#mapping my categorical variables to numbers

val_dict={}

valuess=my_data.neighbourhood_group.unique()

for i in range(len(valuess)):

    val_dict[valuess[i]]=i

val_dict

my_data.neighbourhood_group= my_data.neighbourhood_group.map(val_dict)

data.neighbourhood_group=data.neighbourhood_group.map(val_dict)
hidden_units=(256,128)

host_id=keras.Input(shape=(1,),name='host_id')

neighbourhood_group=keras.Input(shape=(1,),name='neighbourhood_group')

#neighbourhood=keras.Input(shape=(1,),name='neighbourhood')

price=keras.Input(shape=(1,),name='price')

#minimum_nights=keras.Input(shape=(1,),name='minimum_nights')



host_embedded=keras.layers.Embedding(len(my_data.host_id.unique())+1,32,input_length=1)(host_id)

price_embedded=keras.layers.Embedding(len(my_data.price.unique())+1,32,input_length=1)(price)

neighbourhood_embedded=keras.layers.Embedding(len(my_data.neighbourhood_group.unique())+1,32,input_length=1)(neighbourhood_group)



lis=[host_embedded,price_embedded,neighbourhood_embedded]



concatenated=keras.layers.Concatenate()(lis)

#concatenated=keras.layers.Concatenate()([host_embedded,price_embedded])



out = keras.layers.Flatten()(concatenated)



for n_hidden in hidden_units:

    out = keras.layers.Dense(n_hidden, activation='relu')(out)

    

out = keras.layers.Dense(3, activation='softmax', name='prediction')(out)



model = keras.Model(

    inputs = [ host_id,price,neighbourhood_group],

    outputs = out,

)

model.summary(line_length=88)

model.compile(

    # Technical note: when using embedding layers, I highly recommend using one of the optimizers

    # found  in tf.train: https://www.tensorflow.org/api_guides/python/train#Optimizers

    # Passing in a string like 'adam' or 'SGD' will load one of keras's optimizers (found under 

    # tf.keras.optimizers). They seem to be much slower on problems like this, because they

    # don't efficiently handle sparse gradient updates.

    tf.train.AdamOptimizer(0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)




history = model.fit(

    [my_data.host_id, my_data.price,my_data.neighbourhood_group],

    y,

    batch_size=64,

    epochs=10,

    verbose=2,

    validation_split=.10,

);
history.history.keys()
from sklearn.metrics import confusion_matrix

import numpy as np

y_pred=y_test_1=np.argmax( model.predict([data.iloc[40000:,2],data.iloc[40000:,9],data.iloc[40000:,4]]) , axis=1) 

matrix = confusion_matrix(y_pred= y_pred,y_true= data.iloc[40000:,8])

print(matrix)
data.head(1)