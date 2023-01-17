import numpy as np
import pandas as pd
import re
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')
data=pd.concat([train_data[['question1','question2']],test_data[['question1','question2']]],axis=0)
data.head()
len(train_data)+len(test_data),len(data)
stop_words=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def remove_punctuation(sents):
    return re.sub(r'[\W]',' ',str(sents).lower())

def remove_stopwords(sents):
    words=sents.split()
    result=[word.lower() for word in words if word.lower() not in stop_words]
    return " ".join(result)

def clean_data():
    data['cleaned_question_1']=data.question1.apply(remove_punctuation)
    data['cleaned_question_2']=data.question2.apply(remove_punctuation)
    
    data.cleaned_question_1=data.cleaned_question_1.apply(remove_stopwords)
    data.cleaned_question_2=data.cleaned_question_2.apply(remove_stopwords)
    
clean_data()
data.head()
text=np.hstack([data.cleaned_question_1,data.cleaned_question_2])
tokenizer=Tokenizer()
tokenizer.fit_on_texts(text)
data['input1']=tokenizer.texts_to_sequences(data.cleaned_question_1)
data['input2']=tokenizer.texts_to_sequences(data.cleaned_question_2)
data.head()
MAX_LENGTH=80
MAX_TOKEN = np.max([np.max(data.input1.max()),np.max(data.input2.max())])
MAX_LENGTH,MAX_TOKEN
train=data[:len(train_data)-40000]
valid=data[len(train_data)-40000:len(train_data)]
test=data[len(train_data):]
train_x1 = pad_sequences(train.input1, maxlen=MAX_LENGTH)
train_x2 = pad_sequences(train.input2, maxlen=MAX_LENGTH)
train_y=train_data.is_duplicate[:len(train)]

valid_x1 = pad_sequences(valid.input1, maxlen=MAX_LENGTH)
valid_x2 = pad_sequences(valid.input2, maxlen=MAX_LENGTH)
valid_y=train_data.is_duplicate[len(train):]

# test_x1 = pad_sequences(test.input1, maxlen=MAX_LENGTH)
# test_x2 = pad_sequences(test.input2, maxlen=MAX_LENGTH)


from keras.models import model_from_json,Model,Sequential
from keras.layers import Input,Dropout,Dense,LSTM,Embedding,Merge
from keras.optimizers import Adam


A1 = Input(shape=[MAX_LENGTH], name="in1")
B1 = Embedding(MAX_TOKEN, 128)(A1)
C1 = LSTM(90) (B1)
D1 = Dropout(0.6) (Dense(128, activation='relu') (C1))
E1 = Dropout(0.4) (Dense(32, activation='relu') (D1))
F1 = Dropout(0.4) (Dense(32, activation='relu') (E1))
output1 = Dense(64, activation="relu") (F1)

A2 = Input(shape=[MAX_LENGTH], name="in2")
B2 = Embedding(MAX_TOKEN, 128)(A2)
C2 = LSTM(90) (B2)
D2 = Dropout(0.6) (Dense(128, activation='relu') (C2))
E2 = Dropout(0.4) (Dense(32, activation='relu') (D2))
F2 = Dropout(0.4) (Dense(32, activation='relu') (E2))
output2 = Dense(64, activation="relu") (F2)

Model1=Model(A1,output1)
Model2=Model(A2,output2)
print(Model1.summary())
print(Model2.summary())
merged = Merge([Model1,Model2], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(28, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

ad=Adam(0.001,decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=ad, metrics=['accuracy'])
model.fit([train_x1,train_x2], train_y,validation_data = ([valid_x1,valid_x2], valid_y),
            epochs=5, batch_size=128)

_model=model.to_json()
with open("Model.json","w") as json_file:
    json_file.write(_model)
model.save_weights("weights.h5")
