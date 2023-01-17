from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
print ("Read Dataset ... ")
def read_dataset(path):
	return json.load(open(path))
from random import shuffle
train = read_dataset('../input/train.json')
shuffle(train)
shuffle(train)
test = read_dataset('../input/test.json')
print ("Prepare text data of Train and Test ... ")
def generate_text(data):
    DATA=[doc['ingredients'][:] for doc in data]
    return DATA
train_text = generate_text(train)
test_text = generate_text(test)
testids = [i['id'] for i in test ]
train_target = [doc['cuisine'] for doc in train]
train_target[0:3]
print(len(train_text))
print(testids[0:3])
C=sorted(list(set(train_target)))
Map= dict((c, i) for i, c in enumerate(C))
print(Map)

train_target = [Map[i] for i in train_target]
train_target[0:3]
feachers_set= set()
for i in train_text:
    for j in i:
        feachers_set.add(j)
for i in test_text:
    for j in i:
        feachers_set.add(j)
feachers_set = list(feachers_set )
print(len(feachers_set))
X_train =[]
X_test = []

for j in train_text:
    x=[0 for i in range(len(feachers_set))]
    for i in j:
        k= feachers_set.index(i)
        if k>=0:
            x[k]=1
        else :
            print("error")
    X_train.append(x)
#print(X_train[1])    
for j in test_text:
    x=[0 for i in range(len(feachers_set))]
    for i in j:
        k= feachers_set.index(i)
        if k>=0:
            x[k]=1
        else :
            print("error")
    X_test.append(x)
length1 = len(X_train)
length2=  len(X_test)
xlines=X_train[:]+X_test[:]
length3=len(xlines)
print(length1, length2, length3)
CHIX=4000
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
from sklearn import decomposition
LENGTH= 3000

pca = decomposition.PCA(n_components=LENGTH)
pca.fit(xlines)
xlines = pca.transform(xlines)
xlines=xlines.tolist()
print(len(xlines[0]))
#print(xlines[0])
print(xlines[0][1:10])
x_train = xlines[0: length1]
x_test = xlines[length1:]
print(len(x_train), len(x_test))

from keras.utils import np_utils
import numpy as np
from sklearn.utils import class_weight
Y=np_utils.to_categorical(train_target)
X=np.array(x_train)
print(len(Y[0]),Y[0])

    
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout ,ActivityRegularization,LeakyReLU
from keras.regularizers import l1_l2,l1

model = Sequential()
model.add(Dense(1000, activation='relu',input_shape=(LENGTH,)))
model.add(Dropout(0.3))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
#model.add(LeakyReLU(alpha=.1))
#model.add(ActivityRegularization(l2=0.1))
#model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
#model.add(LeakyReLU(alpha=.1))
#model.add(ActivityRegularization(l2=0.1))
#model.add(Dropout(0.2))
model.add(Dense(len(Y[0]), activation='softmax'))
model.compile(optimizer='nadam',
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
revmap={}
for i in Map.keys():
    revmap[Map[i]]=i
print(revmap)
XX=np.array(x_test)
Ans= model.predict(XX)
Ans=[np.argmax(pred) for pred in Ans]
print(Ans[1])
actcual =[]
for i in Ans:
    actcual.append(revmap[i])
import pandas as pd
sub = pd.DataFrame({'id': testids, 'cuisine': actcual}, columns=['id', 'cuisine'])
sub.to_csv('svm_output.csv', index=False)

