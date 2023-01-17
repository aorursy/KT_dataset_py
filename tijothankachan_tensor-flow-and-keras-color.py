#importing necessary libraries

import tensorflow as tf

from tensorflow import keras

import numpy as np

import pandas as pd
#loading data

train_df = pd.read_csv('../input/colors/train.csv')

np.random.shuffle(train_df.values)



train_df.head()


model = keras.Sequential([                                           

	keras.layers.Dense(32, input_shape=(2,), activation='relu'),     #passing number of neurons in hidden layer ,input layer ,output layer

    keras.layers.Dense(32, activation='relu'),

	keras.layers.Dense(6, activation='sigmoid')])



model.compile(optimizer='adam',                                      #compiling

	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),

	          metrics=['accuracy'])

#input and output data to be trained

x=train_df[['x','y']]                                         

train_df['color1']=train_df.color.factorize()[0]
#fitting

model.fit(x, train_df.color1, batch_size=8, epochs=10)
#loading test data set

test_df = pd.read_csv('../input/colors/test.csv')

test_x=test_df[['x','y']]                                      



test_df['color1']=test_df.color.factorize()[0]
#evaluation

print("EVALUATION")

model.evaluate(test_x, test_df.color1)



print('prediction',np.round(model.predict(np.array([[0,3]]))))
# #passing column

# train_df = pd.read_csv('../input/colors/train.csv')

# np.random.shuffle(train_df.values)



# print(train_df.head())



# model = keras.Sequential([

# 	keras.layers.Dense(32, input_shape=(2,), activation='relu'),

#     keras.layers.Dense(32, activation='relu'),

# 	keras.layers.Dense(6, activation='sigmoid')])



# model.compile(optimizer='adam', 

# 	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),

# 	          metrics=['accuracy'])



# # x = np.column_stack((train_df.x.values, train_df.y.values))

# x=train_df[['x','y']]                                          #passing column

# train_df['color1']=train_df.color.factorize()[0]



# model.fit(x, train_df.color1, batch_size=8, epochs=10)



# test_df = pd.read_csv('../input/colors/test.csv')

# # test_x = np.column_stack((test_df.x.values, test_df.y.values))

# test_x=test_df[['x','y']]                                      #passing column



# test_df['color1']=test_df.color.factorize()[0]

# print("EVALUATION")

# model.evaluate(test_x, test_df.color1)



# print('prediction',np.round(model.predict(np.array([[0,3]]))))
#converting back to color

print('prediction',np.round(model.predict(np.array([[0,3]]))))

a=np.round(model.predict(np.array([[0,3]])))

color_dict={0:'red', 1:'blue', 2:'green', 3:'teal', 4:'orange', 5:'purple'}

for i in range(0,6):

        if a[0][i]==1:

            print(color_dict[i])