import numpy as np

import pandas as pd

import seaborn as sns

import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn import svm

%matplotlib inline

import tensorflow as tf

from sklearn.model_selection import StratifiedKFold

from tensorflow import keras as kr



print("TF version ", tf.__version__)



print(os.listdir("../input"))



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



Y_train = train_df['target'] # keep labels

X_train = train_df.iloc[:,2:train_df.shape[1]].values # dataframe to numpy array

X_test = test_df.iloc[:,1:test_df.shape[1]].values # dataframe to numpy array



g = sns.countplot(train_df['target']) # label count
mmscaler = MinMaxScaler()

X_train = mmscaler.fit_transform(X_train)

X_train.shape



# pca = PCA(n_components=12)

# X_train = pca.fit_transform(X_train)  

# X_train.shape
kmodel = kr.models.Sequential()

kmodel.add(kr.layers.Dense(32, input_dim=np.size(X_train,1), activation='relu'))

kmodel.add(kr.layers.Dense(32, activation='relu'))

kmodel.add(kr.layers.Dense(16, activation='relu'))

kmodel.add(kr.layers.Dense(1, activation='sigmoid'))

# model.summary()



kmodel.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')





BATCH_SIZE = 32

EPOCHS = 50

LOGS = 0

VALIDATION_SPLIT = 0.1



history = kmodel.fit(X_train,Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=LOGS, shuffle=True, class_weight=None, sample_weight=None) # train

""" PLOT TRAIN HISTORY """

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
from sklearn.linear_model import LogisticRegression



import warnings

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)





# mmscaler = MinMaxScaler()

X_test = mmscaler.fit_transform(X_test)





model = LogisticRegression(C=.9, penalty='l1', solver='saga',max_iter = 100, warm_start=True)



_ = model.fit(X_train,Y_train) 

predicted = model.predict(X_test)



# seed = 8

# np.random.seed(seed)

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)



# # for name, model in models:

# cvscores = []

# i = 0

# for train, test in kfold.split(X_train, Y_train):



#     _ = model.fit(X_train[train],Y_train[train]) # train

#     predicted = model.predict(X_train[test])

#     i += 1

#     cvscores.append(np.mean(predicted == Y_train[test]) * 100)

#     #print('iter ',str(i))



# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))



predicted.shape



out = pd.Series(predicted,name="target")



out = pd.concat([test_df['id'],out],axis = 1)

#g = sns.countplot(out['target']) # label count

out.to_csv('dont_overfit_it.csv',index=False, sep=',')