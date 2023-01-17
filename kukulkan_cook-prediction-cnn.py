# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_json('../input/train.json')
display(train_df)
test_df = pd.read_json('../input/test.json')
display(test_df)
cuisine_list = []
ingredient_list = []
ingredient_list2 = []
vocabulary = {}
vocabulary_inv = {}
ingredient_num_list = []
ingredient_len_list = []
vocabulary["PADDING"] = 0
vocabulary_inv[0] = "PADDING"
for i,row in train_df.iterrows():
    cuisine = row[0]
    if(not cuisine in cuisine_list):
        cuisine_list.append(cuisine)
    ingredient_list = row[2]
    ingredient_num_list.append(len(ingredient_list))
    for ingredient in ingredient_list:
        ingredient_len_list.append(len(ingredient.split()))
        if(not ingredient.lower() in ingredient_list):
            ingredient_list.append(ingredient.lower())
        for ingredient_part in ingredient.split():
            if(not ingredient_part.lower() in ingredient_list2):
                ingredient_list2.append(ingredient_part.lower())
                vocabulary[ingredient_part.lower()] = len(vocabulary)
                vocabulary_inv[len(vocabulary_inv)] = ingredient_part.lower()
#===
for i,row in test_df.iterrows():
    ingredient_list = row[1]
    ingredient_num_list.append(len(ingredient_list))
    for ingredient in ingredient_list:
        ingredient_len_list.append(len(ingredient.split()))
        if(not ingredient in ingredient_list):
            print("total =",ingredient_list)
            print(ingredient," is not contained!")
            ingredient_list.append(ingredient)
        for ingredient_part in ingredient.split():
                if(not ingredient_part.lower() in ingredient_list2):
                    print(ingredient_part," is not contained as part!")
                    ingredient_list2.append(ingredient_part.lower())
                    vocabulary[ingredient_part.lower()] = len(vocabulary)
                    vocabulary_inv[len(vocabulary_inv)] = ingredient_part.lower()
assert len(cuisine_list)==len(set(cuisine_list)),"cuisine_list duplicated!"
assert len(ingredient_list)==len(set(ingredient_list)),"ingredient_list duplicated!"
assert len(ingredient_list2)==len(set(ingredient_list2)),"ingredient_list2 duplicated!"
print("cuisine_set =",cuisine_list)
print("max ingredient num =",max(ingredient_num_list))
max_ingredient_num = max(ingredient_num_list)
print("max ingredient len =",max(ingredient_len_list))
max_ingredient_len = max(ingredient_len_list)
trainX = []
trainY = []
for i,row in train_df.iterrows():
    cuisine = row[0]
    cuisine_label = np.zeros(len(cuisine_list))
    cuisine_label[cuisine_list.index(cuisine)] = 1
    trainY.append(cuisine_label)
    #===
    ingredient_list = row[2]
    ingredient_token_list = []
    for j in range(max_ingredient_num):
        ingredient_part_token_list = []
        if(j<len(ingredient_list)):
            ingredient = ingredient_list[j]
            ingredient_part_split = ingredient.split()
            for k in range(max_ingredient_len):
                if(k<len(ingredient_part_split)):
                    ingredient_part_token_list.append(vocabulary[ingredient_part_split[k].lower()])
                else:
                    ingredient_part_token_list.append(0)
        else:
            for k in range(max_ingredient_len):
                ingredient_part_token_list.append(0)
        ingredient_token_list.append(ingredient_part_token_list)
    #==
    trainX.append(ingredient_token_list)
#==
testX = []
for i,row in test_df.iterrows():
    ingredient_list = row[1]
    ingredient_token_list = []
    for j in range(max_ingredient_num):
        ingredient_part_token_list = []
        if(j<len(ingredient_list)):
            ingredient = ingredient_list[j]
            ingredient_part_split = ingredient.split()
            for k in range(max_ingredient_len):
                if(k<len(ingredient_part_split)):
                    ingredient_part_token_list.append(vocabulary[ingredient_part_split[k].lower()])
                else:
                    ingredient_part_token_list.append(0)
        else:
            for k in range(max_ingredient_len):
                ingredient_part_token_list.append(0)
        ingredient_token_list.append(ingredient_part_token_list)
    #==
    testX.append(ingredient_token_list)
#==
trainX = np.array(trainX)
print(np.shape(trainX))
trainY = np.array(trainY)
print(np.shape(trainY))
testX = np.array(testX)
print(np.shape(testX))
from sklearn.model_selection import train_test_split
#(trainX, evalX, trainY, evalY) = train_test_split(trainX,trainY, test_size=0.1, shuffle=False)
trainX2 = np.split(trainX, max_ingredient_num, axis=1)
trainX2 =  np.reshape(trainX2, (len(trainX2),len(trainX2[0]),max_ingredient_len))
#evalX2 = np.split(evalX, max_ingredient_num, axis=1)
#evalX2 =  np.reshape(evalX2, (len(evalX2),len(evalX2[0]),max_ingredient_len))
testX2 = np.split(testX, max_ingredient_num, axis=1)
testX2 =  np.reshape(testX2, (len(testX2),len(testX2[0]),max_ingredient_len))
print(np.shape(trainX),np.shape(testX),np.shape(trainX2),np.shape(testX2))
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, LSTM,Bidirectional
from keras.layers.merge import Concatenate
#==
embedding_dim = 400
filter_sizes = (4, 8)
num_filters = 100
hidden_dims = 300
#==
input_shape = (max_ingredient_len,)
input_blocks = []
z_blocks = []
Converter = Embedding(len(vocabulary), embedding_dim, input_length=max_ingredient_len,mask_zero=True)
Convo = Convolution1D(filters=num_filters*2,kernel_size=filter_sizes[0],padding="valid",activation="relu",strides=1)
Pool = MaxPooling1D(pool_size=2)
Drop1 = Dropout(0.5)
Bi_LSTM = Bidirectional(LSTM(num_filters,activation='softmax',recurrent_activation='tanh',use_bias=True,recurrent_dropout=0.0,return_sequences=False))
Dense1 = Dense(20)
for i in range(max_ingredient_num):
    model_input = Input(shape=input_shape)
    input_blocks.append(model_input)
    z = Converter(model_input)
    z = Drop1(z)
    #conv = Convo(z)
    #conv = Pool(conv)
    pressed = Bi_LSTM(z)
    dense = Dense1(pressed)
    z_blocks.append(dense)
concatted = Concatenate()(z_blocks)
concatted = Dropout(0.5)(concatted)
concatted = Dense(hidden_dims)(concatted)
model_output = Dense(len(cuisine_list),activation="softmax")(concatted)
model = Model(input_blocks,model_output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
import keras
batch_size = 256
num_epochs = 50
callbacks = []
callbacks.append(keras.callbacks.EarlyStopping('loss', min_delta=1e-3, patience=1))
trainX2_list = []
for train in trainX2:
    trainX2_list.append(train)
model.fit(trainX2_list,trainY,batch_size=batch_size, epochs=num_epochs,validation_split=0.25, verbose=1,callbacks=callbacks)
from datetime import datetime
now_time = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
#
testX2_list = []
for test in testX2:
    testX2_list.append(test)
predicted = model.predict(testX2_list)
test_id = test_df.id
assert len(predicted)==len(test_id),"Prediction num does not match!"
id_list = []
pred_cuisine_list = []
for food_id,pred in zip(test_id,predicted):
    id_list.append(food_id)
    max_index = pred.argmax()
    pred_cuisine = cuisine_list[max_index]
    pred_cuisine_list.append(pred_cuisine)
from collections import OrderedDict
#pred_dict = {"id":id_list,"cuisine":pred_cuisine_list}
pred_dict = OrderedDict([("id",id_list), ("cuisine",pred_cuisine_list)])
pred_df = pd.DataFrame.from_dict(pred_dict)
pred_df.to_csv("submission"+now_time+".csv",index=False)