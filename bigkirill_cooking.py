import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation,Dropout
from keras.layers import LSTM
from keras.preprocessing import sequence
import keras as keras
import numpy as np
data=pd.read_json('../input/train.json')
cat_dict=[]
data_dict=[]
data=data.drop(['id'],axis=1)
top_len=0
for x in range(0,len(data)):
    l=len(data['ingredients'][x])
    if (l>top_len):
        top_len=l
m=[]
for x in range(0,len(data)):
    k=''
    for f in data['ingredients'][x]:
        k+=f.lower()+' '
    m.append(k)
data['ingredients']=m
cuisine_dic_t=[]
for x in range(0,len(data)):
    if data['cuisine'][x] not in cuisine_dic_t:
        cuisine_dic_t.append(data['cuisine'][x])
cuisine_dic=pd.DataFrame(data=cuisine_dic_t)
cuisine=[]
for x in range(0,len(data)):
    for s in range(0,len(cuisine_dic)):
        if cuisine_dic[0][s]==data['cuisine'][x]:
            k=[cuisine_dic.index[s]]
            cuisine.append(k)
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['ingredients'])

ingridients = tokenizer.texts_to_sequences(data['ingredients'])
X_train=ingridients[:30000]
y_train=cuisine[:30000]
X_test=ingridients[30000:]
y_test=cuisine[30000:]
X_train = tokenizer.sequences_to_matrix(X_train,mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test,mode='binary')
y_train = keras.utils.to_categorical(y_train,len(cuisine_dic))
y_test = keras.utils.to_categorical(y_test,len(cuisine_dic))
model = Sequential()
model.add(Dense(2048, input_shape=(len(tokenizer.word_index)+1,)))
model.add(Activation('relu'))
model.add(Dropout(0.7))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(cuisine_dic)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(X_test, y_test))
model.evaluate(X_test, y_test,
                       batch_size=32, verbose=1)
test_data=pd.read_json('../input/test.json')
m=[]
for x in range(0,len(test_data)):
    k=''
    for f in test_data['ingredients'][x]:
        k+=f.lower()+' '
    m.append(k)
test_data['ingredients']=m
ingridients_test = tokenizer.texts_to_sequences(test_data['ingredients'])
ingridients_test=tokenizer.sequences_to_matrix(ingridients_test,mode='binary')
s=model.predict_classes(ingridients_test)
res_t=[]
for x in s:
    res_t.append(cuisine_dic[0][x])
test_data['ingredients']=res_t
test_data=test_data.rename(index=str, columns={'ingredients': "cuisine"})
test_data.to_csv('result.csv', index=False)
