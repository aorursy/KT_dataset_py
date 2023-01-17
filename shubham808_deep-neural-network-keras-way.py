

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest

from sklearn.metrics import accuracy_score

from sklearn.grid_search import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score,KFold

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation,Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils



from keras import regularizers
data = pd.read_csv("../input/train.csv").astype('float_')

test = pd.read_csv("../input/test.csv").astype('float_')

test = test.values

print(data.keys())

print(data.shape)
labels = data['label']

data.drop('label',axis=1,inplace=True)

features=np.array(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25,

                                                                               random_state=42)
scale = np.max(features_train)

features_train /= scale

features_test /= scale

test/=scale

mean = np.std(features_train)

features_train -= mean

features_test -= mean

test-=mean


pca = PCA(n_components=2).fit(features_train)

reduced_features_train = pca.transform(features_train)

reduced_features_test = pca.transform(features_test)

reduced_test = pca.transform(test)
clf=SVC()

clf.fit(reduced_features_train,labels_train)

#Validating by KFold Cross Validation

print(cross_val_score(clf,features,labels,cv=4))
pipeline = Pipeline(steps=[("clf",SVC())])

#using grid earchCV on SVM

param_grid = [

  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},

  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},

 ]

#rbf kernel is default so we as well try with linear too

clf = GridSearchCV(pipeline, param_grid)

clf.fit(reduced_features_train,labels_train)

print(clf.best_score_)

print(clf.best_estimator_)
k_best.fit(features_train, labels_train)

scores = k_best.scores_

i=0



nan_scores = [x for x in scores if x>0]

nan_scores = np.array(nan_scores)

print(min(nan_scores))



features_list=[]

for i in range(len(scores)):

    if scores[i]>min(nan_scores):

        features_list.append(i)

print(features_list)


labels_train = np_utils.to_categorical(labels_train)

labels_test = np_utils.to_categorical(labels_test)
# fix random seed for reproducibility

seed = 43

np.random.seed(seed)

#Set input dimension

input_dim = features_train.shape[1]
model = Sequential()

model.add(Dense(512, input_dim=input_dim,kernel_regularizer=regularizers.l2(0.0001)))

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(128,kernel_regularizer=regularizers.l2(0.0001)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes))

model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("Training...")

model.fit(features_train, labels_train, validation_split=0.1, verbose=1)

print(model.evaluate(features_test,labels_test,batch_size=batch_size,verbose=0))

print("Generating test predictions...")

preds = model.predict_classes(test, verbose=0)


features_train = features_train.reshape(features_train.shape[0],28,28,1)

features_test = features_test.reshape(features_test.shape[0],28,28,1)
#define our model

model=Sequential()

#declare input layer use tensorflow backend meaning depth comes at the end

model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,1)))

print(model.output_shape)



model.add(Convolution2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#conver to 1D by flatten

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



model.fit(features_train,labels_train,validation_split=0.1)

score = model.evaluate(features_test,labels_test,verbose=0)
test = test.reshape(test.shape[0],28,28,1)



pred = model.predict_classes(test,verbose=0)



pred = pd.DataFrame(pred)