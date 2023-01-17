import pandas as pd

import numpy as np 

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import normalize as normalizer

from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/Churn_Modelling.csv')

df.head()
filtered_df = df.drop(labels=['RowNumber','CustomerId','Surname'],axis=1,inplace=True)
y=df['Exited']

X=df.drop(labels='Exited',axis=1)
encoded_y=y.to_numpy().astype('float64')
encoded_Georaphy=pd.get_dummies(X['Geography'])
encoded_Gender=pd.get_dummies(X['Gender'])
X.drop(labels=['Geography','Gender'],axis=1,inplace=True)
frames = [X,encoded_Georaphy,encoded_Gender]
engg_datgaframe=pd.concat(frames,sort=False,axis=1)
numpy_array=engg_datgaframe.to_numpy()
normalized_dataset=normalizer(numpy_array,axis=1)
#Train Test Spliting
X_train, X_test, y_train, y_test = train_test_split(normalized_dataset, encoded_y, test_size=0.30, random_state=10)
print(f'''

Traing Size={X_train.shape}

Testing Size={X_test.shape}

Labels_train Size={y_train.shape}

Labels_test Size={y_test.shape}

''')
# Model architecture
network = tf.keras.Sequential()

network.add(tf.keras.layers.Dense(1024,activation='relu',input_shape=(13,)))

network.add(tf.keras.layers.Dense(512,activation='relu'))

network.add(tf.keras.layers.Dense(1,activation='sigmoid'))

network.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
network.fit(X_train,y_train,epochs=20,batch_size=20,validation_split=0.3,verbose=1)
network.evaluate(X_test,y_test,verbose=0)