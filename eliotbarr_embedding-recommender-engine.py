%matplotlib inline



import matplotlib.pyplot as plt

import numpy as np

import os.path as op

import pandas as pd



from  keras.layers  import Input, Embedding, Flatten, merge, Dense, Dropout, Lambda

from keras.models import Model

import keras.backend as K

import keras



import tensorflow as tf

from  sklearn.model_selection  import train_test_split

from sklearn.preprocessing import LabelEncoder

import os

print(os.listdir("../input"))
notes = pd.read_csv('../input/rating_final.csv')
notes.head()
notes['userID'] = LabelEncoder().fit_transform(notes['userID']) 
notes ['rating'].describe()
max_user_id  = notes['userID'].max()

max_user_id
max_item_id = notes['placeID'].max()

max_item_id
num_classes = 3



ratings_train, ratings_test = train_test_split(

    notes, test_size=0.2, random_state=0)



user_id_train = ratings_train['userID']

item_id_train = ratings_train['placeID']

rating_train = ratings_train['rating']



user_id_test = ratings_test['userID']

item_id_test = ratings_test['placeID']

rating_test = ratings_test['rating']



rating_train = keras.utils.to_categorical(rating_train, num_classes)

rating_test = keras.utils.to_categorical(rating_test, num_classes)
# For each sample we input the integer identifiers

# of a single user and a single item

user_id_input = Input(shape=[1], name='user')

item_id_input = Input(shape=[1], name='item')



embedding_size = 30

user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,

                           input_length=1, name='user_embedding')(user_id_input)

item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,

                           input_length=1, name='item_embedding')(item_id_input)



# reshape from shape: (batch_size, input_length, embedding_size)

# to shape: (batch_size, input_length * embedding_size) which is

# equal to shape: (batch_size, embedding_size)

user_vecs = Flatten()(user_embedding)

item_vecs = Flatten()(item_embedding)



input_vec = merge([user_vecs, item_vecs], mode='concat')

y = Dense(3, activation='softmax')(input_vec)



model = Model(input=[user_id_input, item_id_input], output=y)

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy)
%%time



# Training the model

history = model.fit([user_id_train, item_id_train], rating_train,

                    batch_size=32, nb_epoch=25, validation_split=0.1,

                    shuffle=True, verbose=2)
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='validation')

plt.ylim(0, 2)

plt.legend(loc='best')

plt.title('Loss');
from  sklearn.metrics  import mean_squared_error

from sklearn.metrics import mean_absolute_error



test_preds = model.predict([user_id_test, item_id_test])

print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))

print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))
model.evaluate([user_id_test, item_id_test], y=rating_test)