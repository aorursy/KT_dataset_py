from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import json
from random import shuffle
# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
	return json.load(open(path)) 
train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')

shuffle(train)
shuffle(train)
# Text Data Features
print ("Prepare text data of Train and Test ... ")
def generate_text(data):
	text_data = [" ".join(doc['ingredients']).lower() for doc in data]
	return text_data 

train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]

# Feature Engineering 
print ("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True)
def tfidf_features(txt, flag):
    if flag == "train":
    	x = tfidf.fit_transform(txt)
    else:
	    x = tfidf.transform(txt)
    x = x.astype('float16')
    return x 

X = tfidf_features(train_text, flag="train")
X_test = tfidf_features(test_text, flag="test")

# Label Encoding - Target 
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)
print ("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True)
def tfidf_features(txt, flag):
    if flag == "train":
    	x = tfidf.fit_transform(txt)
    else:
	    x = tfidf.transform(txt)
    #x = x.astype('float16')
    return x 

X = tfidf_features(train_text, flag="train")
X_test = tfidf_features(test_text, flag="test")

# Label Encoding - Target 

X=X.toarray().tolist()
X_test = X_test.toarray().tolist()
print(y[1:3])
'''length1 = len(X)
length2=  len(X_test)
xlines=X[:]+X_test[:]
length3=len(xlines)
print(length1, length2, length3)
LENGTH=2000
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif
from sklearn import decomposition
f=SelectKBest(chi2, k=LENGTH)
X = f.fit_transform(X,y)
mast=f.get_support()

feachers=[]
i=0
for bool in mast:
    if bool:
        feachers.append(i)
    i+=1

X_test= [[j[i] for i in feachers] for j in X_test]'''
'''LENGTH= 2000
pca = decomposition.PCA(n_components=LENGTH)
pca.fit(xlines)
xlines = pca.transform(xlines)
xlines=xlines.tolist()
print(len(xlines[0]))
X = xlines[0: length1]
X_test = xlines[length1:]
print(len(X), len(X_test))'''
from keras.utils import np_utils
import numpy as np
from sklearn.utils import class_weight
Y=y#np_utils.to_categorical(y)

class_weight_ = class_weight.compute_class_weight('balanced',
                                                 np.unique(y),
                                                 y)
class_weights={}
for i in range(len(class_weight_)):
    class_weights[i]=100/class_weight_[i]
X=np.array(X)
X_test=np.array(X_test)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout ,ActivityRegularization,LeakyReLU
from keras.regularizers import l1_l2,l1
from keras import optimizers
from keras import regularizers
model = Sequential()
model.add(Dense(500, activation='relu',input_shape=(3010,)))
model.add(Dropout(0.25))
##model.add(Dense(500, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
#model.add(LeakyReLU(alpha=.2))
#model.add(Dense(250, activation='linear'))
#model.add(LeakyReLU(alpha=.2))
#model.add(ActivityRegularization(l2=0.1))

#model.add(LeakyReLU(alpha=.2))
#model.add(ActivityRegularization(l2=0.1))
#model.add(Dense(100, activation='relu'))
#model.add(LeakyReLU(alpha=.2))
#model.add(ActivityRegularization(l2=0.1))
model.add(Dropout(0.2))
model.add(Dense(len(Y[0]), activation='softmax'))
sgd = optimizers.SGD(lr=.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='nadam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
model.summary()
'''from keras.callbacks import ModelCheckpoint
path_model='model_simple_keras_starter.h5' 
checkpointer = ModelCheckpoint('model_simple_keras_starter.h5',monitor='val_acc', verbose=1, save_best_only=True)
model.fit(X,Y,epochs=20, 
            verbose=20,
          batch_size=10,
            validation_data=(X[33000:],Y[33000:]),
            shuffle=True ,
          callbacks=[
                checkpointer,
            ]
         )
callbacks=[
                checkpointer,
            ]'''
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
model = DecisionTreeClassifier(criterion = "gini", random_state = 1000,
                               max_depth=1000, min_samples_leaf=1)
model.fit(X, Y)
#model.load_weights('model_simple_keras_starter.h5')
score = model.score(X[33000:],Y[33000:])
print('Test accuracy:', score)

Ans= model.predict(X_test)
print(Ans[0])
#Ans=[ np.argmax(i) for i in Ans]
Ans=  lb.inverse_transform(Ans)
print ("Generate Submission File ... ")
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': Ans}, columns=['id', 'cuisine'])
sub.to_csv('svm_output.csv', index=False)