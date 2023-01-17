from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import json
from random import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.utils import np_utils
# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
	return json.load(open(path)) 
train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')

shuffle(train)
#shuffle(train)
# Text Data Features
print ("Prepare text data of Train and Test ... ")
def generate_text(data):
	text_data = [" ".join(doc['ingredients']).lower() for doc in data]
	return text_data 

train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]

# Feature Engineering 

feachers_set= set()
for i in train_text:
    for j in i:
        feachers_set.add(j)
for i in test_text:
    for j in i:
        feachers_set.add(j)
feachers_set = list(feachers_set )
voclen=len(feachers_set)
del(feachers_set)


X = [one_hot(d, voclen) for d in train_text ]
X_text = [one_hot(d, voclen) for d in test_text ]
del(train_text)
del(test_text)
maxlen = max([len(i) for i in X])
maxlen1 = max([len(i) for i in X_text])
maxlen= max(maxlen,maxlen1)
X = pad_sequences(X, maxlen=maxlen, padding='post')
X_test = pad_sequences(X_text, maxlen=maxlen, padding='post')
# Label Encoding - Target 
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)
Y=np_utils.to_categorical(y)
print(maxlen, voclen)
#X=X.toarray().tolist()
#X_test = X_test.toarray().tolist()
#print(y[1:3])
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout ,ActivityRegularization,LeakyReLU
from keras.regularizers import l1_l2,l1
from keras import optimizers
from keras import regularizers
model = Sequential()
model.add(Embedding(input_dim=voclen, # 10
                    output_dim=16, 
                    input_length=maxlen))
model.add(Flatten())

model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
#model.add(LeakyReLU(alpha=.2))
#model.add(Dense(250, activation='linear'))
#model.add(LeakyReLU(alpha=.2))
#model.add(ActivityRegularization(l2=0.1))

#model.add(LeakyReLU(alpha=.2))
#model.add(ActivityRegularization(l2=0.1))
model.add(Dense(100, activation='relu'))
#model.add(LeakyReLU(alpha=.2))
#model.add(ActivityRegularization(l2=0.1))
model.add(Dropout(0.3))
model.add(Dense(len(Y[0]), activation='softmax'))
sgd = optimizers.SGD(lr=1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
model.summary()
from keras.callbacks import ModelCheckpoint
path_model='model_simple_keras_starter.h5' 
checkpointer = ModelCheckpoint('model_simple_keras_starter.h5',monitor='val_acc', verbose=1, save_best_only=True)
model.fit(X,Y,epochs=50, 
            verbose=1,
          batch_size=64,
            validation_data=(X[33000:],Y[33000:]),
            shuffle=True,
            callbacks=[
                checkpointer,
            ]
          
         )
model.load_weights('model_simple_keras_starter.h5')
score = model.evaluate(X[33000:],Y[33000:], verbose=0)
print('Test accuracy:', score)
import numpy as np
Ans= model.predict(X_test)
print(Ans[0])
Ans=[ np.argmax(i) for i in Ans]
Ans=  lb.inverse_transform(Ans)
print ("Generate Submission File ... ")
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': Ans}, columns=['id', 'cuisine'])
sub.to_csv('svm_output.csv', index=False)