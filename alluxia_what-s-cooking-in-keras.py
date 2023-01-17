import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.model_selection import train_test_split


%matplotlib inline
# Any results you write to the current directory are saved as output.


train_df=pd.read_json('../input/train.json')
test_df=pd.read_json('../input/test.json')
train_df.head()
test_df.head()
train_df.info()
test_df.info()
len(train_df['cuisine'].unique())
train_df['cuisine'].value_counts().plot(kind='barh')

target=train_df['cuisine']
train=train_df.drop('cuisine',axis=1)
test=test_df
target.head()
t=Tokenizer()
t.fit_on_texts(train['ingredients'])
train_encoded=t.texts_to_matrix(train['ingredients'],mode='tfidf')

cuisines=train_df['cuisine'].unique()
label2index={cuisine:i for i,cuisine in enumerate(cuisines)}
y=[]

for item in target:
    if item in label2index.keys():
        y.append(label2index[item])
y_encoded=to_categorical(y,20)

print(train_encoded.shape)
print(y_encoded.shape)

def build_model():
    model=Sequential()
    model.add(Dense(256,input_shape=[train_encoded.shape[1], ],activation='relu',name='hidden_1'))
    model.add(Dropout(0.4, name='dropout_1'))
    
    #model.add(Dense(64,activation='relu',name='hidden_2'))
    #model.add(Dropout(0.2,name='dropout_2'))
    
    model.add(Dense(20,name='output'))
    
    model.compile(optimizer='adam',
                  loss='categorical_hinge',
                  metrics=['accuracy']
                )
    
    return model
X_train,X_val,y_train,y_val=train_test_split(train_encoded,y_encoded,test_size=0.2,random_state=22)
model=build_model()
model.summary()
monitor=[
    EarlyStopping(monitor='val_loss',patience=5,verbose=1),
    ModelCheckpoint('best-model-0.h5',monitor='val_loss',save_best_only=True,save_weights_only=True)
]

model.fit(X_train,y_train,
         validation_data=(X_val,y_val),
         epochs=100,
         callbacks=monitor,
         batch_size=128)
test_encoded=t.texts_to_matrix(test_df['ingredients'],mode='tfidf')
test_encoded.shape
model.load_weights('best-model-0.h5')
y_pred=model.predict(test_encoded).argmax(axis=1)

results=[]

for i in y_pred:
    for k,v in label2index.items():
        if v==i:
            results.append(k)

results[:10]
submission=pd.DataFrame(list(zip(test_df['id'],results)),columns=['id','cuisine'])
submission.to_csv('submission.csv',header=True,index=False)

submission=pd.read_csv('submission.csv')
submission.head()
