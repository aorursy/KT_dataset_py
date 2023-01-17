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
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential,layers



from tensorflow import feature_column
df=pd.read_csv('/kaggle/input/iris/Iris.csv',index_col='Id')

df
df.dtypes


df['Species']=pd.Categorical(df['Species'])

df['Species']=df.Species.cat.codes
df
df_val= df.sample(frac=0.2, random_state=20)

df_val
df_val.shape
df_train=df.drop(df_val.index)

df_train
df_train.shape
# Function to create Dataset from df

def dfToDataset(df,target,shuffle,batch_size):

    df=df.copy()

    lables=df.pop(target)

    ds=tf.data.Dataset.from_tensor_slices((dict(df),lables))

    if shuffle:

        ds=ds.shuffle(buffer_size=len(df))

    ds=ds.batch(batch_size)

    return ds
train_ds=dfToDataset(df=df_train,target='Species',shuffle=True,batch_size=10)

val_ds=dfToDataset(df=df_val,target='Species',shuffle=True,batch_size=10)
for i in train_ds.take(5):

    for x in i:

        print(x)
feature_colums=[]



for col in df.columns:

    if col=='Species':

        continue

    feature_colums.append(feature_column.numeric_column(col,dtype=tf.float16))

feature_colums
myModel=Sequential()



myModel.add(layers.DenseFeatures(feature_colums))

myModel.add(layers.Dense(32,activation='relu'))

myModel.add(layers.Dense(3,activation='softmax'))
myModel.compile(optimizer='adam',

               loss=keras.losses.binary_crossentropy,

               metrics=['accuracy'])
epochs=100



myModel.fit(train_ds,

           validation_data=val_ds,

           epochs=epochs)
myModel.evaluate(val_ds)
myModel.predict(val_ds)
predications=np.argmax(myModel.predict(val_ds),axis=1)

predications
len(predications)
df.keys()
example={

    'Id':[3],

    'SepalLengthCm':[5.0],

    'SepalWidthCm':[1.2],

    'PetalLengthCm':[3.5],

    'PetalWidthCm':[0.7]

}

example
input_dic={

    name: tf.convert_to_tensor([value]) for name,value in example.items()

}

input_dic
myModel.predict(input_dic)
np.argmax(myModel.predict(input_dic))