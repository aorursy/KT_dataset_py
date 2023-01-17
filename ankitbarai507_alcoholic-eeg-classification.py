# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import scipy.io as sio

test = sio.loadmat('/kaggle/input/alcohol.mat')

test
test['alcohol'].shape
df1=pd.read_csv('/kaggle/input/normal.csv',index_col=0)

df2=pd.read_csv('/kaggle/input/alcohol.csv',index_col=0)

df1=df1.append(df2,ignore_index=True)
df1.info()
df1.head(1)
df=df1

import seaborn as sns

sns.countplot(x='type', data=df)
df.isnull().sum()
df=df.sample(frac=1).reset_index(drop=True)
df.head()
x=df.drop(["type"]  ,axis=1)

x.shape
y = df.loc[:,'type'].values

y.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x)

x = scaler.transform(x)

from keras.utils import to_categorical

y = to_categorical(y)

y
x
# from sklearn.decomposition import PCA

# pca = PCA(n_components=4)

# x = pca.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)
x_train = np.reshape(x_train, (x_train.shape[0],1,x.shape[1]))

x_test = np.reshape(x_test, (x_test.shape[0],1,x.shape[1]))
x_train.shape
import tensorflow as tf

from tensorflow.keras import Sequential



from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import LSTM

tf.keras.backend.clear_session()



model = Sequential()

model.add(LSTM(64, input_shape=(1,2048),activation="relu",return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(32,activation="sigmoid"))

model.add(Dropout(0.2))

#model.add(LSTM(100,return_sequences=True))

#model.add(Dropout(0.2))

#model.add(LSTM(50))

#model.add(Dropout(0.2))

model.add(Dense(2, activation='sigmoid'))

from keras.optimizers import SGD

model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])

model.summary()
history = model.fit(x_train, y_train, epochs = 100, validation_data= (x_test, y_test))

score, acc = model.evaluate(x_test, y_test)
from sklearn.metrics import accuracy_score

pred = model.predict(x_test)

predict_classes = np.argmax(pred,axis=1)

expected_classes = np.argmax(y_test,axis=1)

print(expected_classes.shape)

print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)

print(f"Testing Accuracy: {correct}")
x_complete=x

x_complete = np.reshape(x_complete, (x_complete.shape[0],1,x_complete.shape[1]))

from sklearn.metrics import accuracy_score

pred = model.predict(x_complete)

predict_classes = np.argmax(pred,axis=1)

expected_classes = np.argmax(y,axis=1)

print(expected_classes.shape)

print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)

print(f"Testing Accuracy: {correct}")
x=df.drop(["type"] ,axis=1).values

x.shape

y = df.loc[:,'type'].values

y.shape

y=to_categorical(y)
x.shape
type(x)
y
a = np.zeros(shape=(240,64,32))
for i in range(len(x)):

    a[i]=np.reshape(x[i],(-1,32))

a.shape
x=a
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)



x_train = x_train.reshape(x_train.shape[0], 64, 32, 1)

x_test = x_test.reshape(x_test.shape[0], 64, 32, 1)



from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

from keras.utils import np_utils

# building a linear stack of layers with the sequential model

model = Sequential()

# convolutional layer

model.add(Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu', input_shape=(64,32,1)))

model.add(MaxPool2D(pool_size=(2,2)))



# model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))

# model.add(MaxPool2D(pool_size=(2,2)))



# model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))

# model.add(MaxPool2D(pool_size=(2,2)))



# model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))

# model.add(MaxPool2D(pool_size=(2,2)))



# flatten output of conv

model.add(Flatten())

# hidden layer

model.add(Dense(1000, activation='relu'))

# # hidden layer

# model.add(Dense(512, activation='relu'))

# # hidden layer

# model.add(Dense(256, activation='relu'))

# # hidden layer

# model.add(Dense(64, activation='relu'))

# # hidden layer

# model.add(Dense(16, activation='relu'))

# output layer

model.add(Dense(2, activation='softmax'))



# compiling the sequential model

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')



# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

# mcp_save = ModelCheckpoint('2dcnn_mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')



# training the model for 10 epochs

model.fit(x_train, y_train, batch_size=1, epochs=50, validation_data=(x_test, y_test))#callbacks=[earlyStopping, mcp_save, reduce_lr_loss]
x_complete=x

x_complete = x_complete.reshape(x_complete.shape[0], 64, 32, 1)

from sklearn.metrics import accuracy_score

pred = model.predict(x_complete)

predict_classes = np.argmax(pred,axis=1)

expected_classes = np.argmax(y,axis=1)

print(expected_classes.shape)

print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)

print(f"Testing Accuracy: {correct}")
x_complete.shape
pred = model.predict(x_complete[:48])

predict_classes = np.argmax(pred,axis=1)

expected_classes = np.argmax(y[:48],axis=1)

print(expected_classes.shape)

print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)

print(f"Testing Accuracy: {correct}")
from sklearn.metrics import accuracy_score

pred = model.predict(x_complete[48:96])

predict_classes = np.argmax(pred,axis=1)

expected_classes = np.argmax(y[48:96],axis=1)

print(expected_classes.shape)

print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)

print(f"Testing Accuracy: {correct}")
from sklearn.metrics import accuracy_score

pred = model.predict(x_complete[96:144])

predict_classes = np.argmax(pred,axis=1)

expected_classes = np.argmax(y[96:144],axis=1)

print(expected_classes.shape)

print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)

print(f"Testing Accuracy: {correct}")
from sklearn.metrics import accuracy_score

pred = model.predict(x_complete[144:192])

predict_classes = np.argmax(pred,axis=1)

expected_classes = np.argmax(y[144:192],axis=1)

print(expected_classes.shape)

print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)

print(f"Testing Accuracy: {correct}")
from sklearn.metrics import accuracy_score

pred = model.predict(x_complete[192:])

predict_classes = np.argmax(pred,axis=1)

expected_classes = np.argmax(y[192:],axis=1)

print(expected_classes.shape)

print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)

print(f"Testing Accuracy: {correct}")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

type(X_train)
# X = np.random.randn(4000,270)

# y = np.ones((4000,1))

# y[0:999] = 2

# y[1000:1999] = 3

# y[2000:2999] = 0

x=df.drop(["type"]  ,axis=1).values

y = df.loc[:,'type'].values

y=to_categorical(y)

from keras.layers import Dense, InputLayer, Dropout, Flatten, BatchNormalization, Conv1D, MaxPooling1D

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

num_classes = 2



X_train = X_train.reshape(X_train.shape[0], 2048,1).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 2048, 1).astype('float32') 



model = Sequential()

model.add(Conv1D(filters=32, kernel_size=5, input_shape=(2048, 1)))

model.add(MaxPooling1D(pool_size=5 ))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))





# compiling the sequential model

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')



# training the model for 10 epochs

model.fit(X_train, y_train, batch_size=1, epochs=30, validation_data=(X_test, y_test))
x_complete=x

x_complete = x_complete.reshape(x_complete.shape[0], 2048,1).astype('float32')

from sklearn.metrics import accuracy_score

pred = model.predict(x_complete)

predict_classes = np.argmax(pred,axis=1)

expected_classes = np.argmax(y,axis=1)

print(expected_classes.shape)

print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)

print(f"Testing Accuracy: {correct}")
import scipy.io as sio

temp1 = sio.loadmat('/kaggle/input/alcohol.mat')

alcohol=temp1['alcohol']

alcohol.shape
import scipy.io as sio

temp1 = sio.loadmat('/kaggle/input/normal.mat')

normal=temp1['normal']

normal.shape
y=[1 for i in range(120)]

y.extend([0 for i in range(120)])

y=np.array(y)

y.shape
x=np.concatenate([alcohol,normal])

x.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)
y_train
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.model_selection import cross_val_score

import itertools

from itertools import chain

import time

start = time.time()



clf_rf = RandomForestClassifier()

clf_rf.fit(X_train, y_train)

prediction = clf_rf.predict(X_test)

y_score = clf_rf.predict_proba(X_test)[:,1]

scores = cross_val_score(clf_rf, x, y, cv=5)

end = time.time()



print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report( y_test,prediction))

from sklearn.metrics import roc_auc_score

print('auc= ',roc_auc_score(y_test,y_score))



start = time.time()



clf_et = ExtraTreesClassifier()

clf_et.fit(X_train, y_train)

prediction = clf_et.predict(X_test)

scores = cross_val_score(clf_et, x, y, cv=5)



end = time.time()





print("Extra Trees Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report( y_test,prediction))

start = time.time()



clf_dt = DecisionTreeClassifier()

clf_dt.fit(X_train, y_train)

prediction = clf_dt.predict(X_test)

scores = cross_val_score(clf_dt, x, y, cv=5)

print(confusion_matrix( y_test,prediction))

end = time.time()



print("Dedicion Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report( y_test,prediction))
start = time.time()



clf_rf = RandomForestClassifier()

clf_rf.fit(X_train, y_train)

prediction = clf_rf.predict(x)

y_score = clf_rf.predict_proba(x)[:,1]

scores = cross_val_score(clf_rf, x, y, cv=5)

end = time.time()



print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction, y)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report( y,prediction))

from sklearn.metrics import roc_auc_score

print('auc= ',roc_auc_score(y,y_score))
X=x

from sklearn.svm import SVC, NuSVC, LinearSVC



start = time.time()



clf = SVC(kernel='linear',C=1,gamma=1)

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)





end = time.time()

labels = np.unique(y_test)

# accuracy_all.append(accuracy_score(prediction, y_test))

# cvs_all.append(np.mean(scores))



print("SVC LINEAR Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report(y_test, prediction))



print(confusion_matrix(y_test, prediction,labels=labels))



start = time.time()



clf_svmpoly = SVC(kernel='poly')

clf_svmpoly.fit(X_train, y_train)

prediction = clf_svmpoly.predict(X_test)

scores = cross_val_score(clf_svmpoly, X, y, cv=5)





end = time.time()



# accuracy_all.append(accuracy_score(prediction, y_test))

# cvs_all.append(np.mean(scores))



print("SVC POLY Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report(y_test, prediction))



start = time.time()



clf_svmsigmoid = SVC(kernel='sigmoid',probability=True)

clf_svmsigmoid.fit(X_train, y_train)

prediction = clf_svmsigmoid.predict(X_test)

scores = cross_val_score(clf_svmsigmoid, X, y, cv=5)



end = time.time()



# accuracy_all.append(accuracy_score(prediction, y_test))

# cvs_all.append(np.mean(scores))



print("SVC SIGMOID Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report(y_test, prediction))





start = time.time()



clf_svmrbf = SVC(kernel='rbf',probability=True)

clf_svmrbf.fit(X_train, y_train)

prediction = clf_svmrbf.predict(X_test)

scores = cross_val_score(clf_svmrbf, X, y, cv=5)





end = time.time()



# accuracy_all.append(accuracy_score(prediction, y_test))

# cvs_all.append(np.mean(scores))



print("SVC RBF Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report( y_test,prediction))

start = time.time()



clf_svmnu = NuSVC()

clf_svmnu.fit(X_train, y_train)

prediciton = clf_svmnu.predict(X_test)

scores = cross_val_score(clf_svmnu, X, y, cv=5)

end = time.time()





print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report( y_test,prediction))

start = time.time()



clf_svmlinear = LinearSVC()

clf_svmlinear.fit(X_train, y_train)

prediction = clf_svmlinear.predict(X_test)

scores = cross_val_score(clf_svmlinear, X, y, cv=5)





print("LinearSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report( y_test,prediction))
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

# define models and parameters

model = LogisticRegression()

solvers = ['newton-cg', 'lbfgs', 'liblinear']

penalty = ['l2']

c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search

grid = dict(solver=solvers,penalty=penalty,C=c_values)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X, y)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.linear_model import RidgeClassifier

# define models and parameters

model = RidgeClassifier()

alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# define grid search

grid = dict(alpha=alpha)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X, y)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.neighbors import KNeighborsClassifier

# define models and parameters

model = KNeighborsClassifier()

n_neighbors = range(1, 21, 2)

weights = ['uniform', 'distance']

metric = ['euclidean', 'manhattan', 'minkowski']

# define grid search

grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X, y)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# define model and parameters

model = SVC()

kernel = ['poly', 'rbf', 'sigmoid']

C = [50, 10, 1.0, 0.1, 0.01]

gamma = ['scale']

# define grid search

grid = dict(kernel=kernel,C=C,gamma=gamma)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X, y)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier()

n_estimators = [1,2,3,4,5,8,10, 100, 1000]

# define grid search

grid = dict(n_estimators=n_estimators)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X, y)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

n_estimators = [10, 100, 1000]

max_features = ['sqrt', 'log2']

# define grid search

grid = dict(n_estimators=n_estimators,max_features=max_features)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X, y)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# from sklearn.ensemble import GradientBoostingClassifier

# # define models and parameters

# model = GradientBoostingClassifier()

# n_estimators = [10, 100, 1000]

# learning_rate = [0.001, 0.01, 0.1]

# subsample = [0.5, 0.7, 1.0]

# max_depth = [3, 7, 9]

# # define grid search

# grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)

# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

# grid_result = grid_search.fit(X, y)

# # summarize results

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']

# stds = grid_result.cv_results_['std_test_score']

# params = grid_result.cv_results_['params']

# for mean, stdev, param in zip(means, stds, params):

#     print("%f (%f) with: %r" % (mean, stdev, param))
svc=SVC(kernel='poly')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score:')

print(accuracy_score(y_test,y_pred))
svc=SVC(kernel='poly')

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation

print(scores)
from sklearn.naive_bayes import GaussianNB



start = time.time()



clf_gnb = GaussianNB()

clf_gnb.fit(X_train, y_train)

prediction = clf_gnb.predict(X_test)

scores = cross_val_score(clf_gnb, X, y, cv=5)

end = time.time()



print("Gaussian NB Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report( y_test,prediction))
start = time.time()



clf_svmpoly = SVC(kernel='poly')

clf_svmpoly.fit(X_train[:,0].reshape(-1, 1), y_train)

prediction = clf_svmpoly.predict(X_test[:,0].reshape(-1, 1))

scores = cross_val_score(clf_svmpoly, X[:,0].reshape(-1, 1), y, cv=5)





end = time.time()



# accuracy_all.append(accuracy_score(prediction, y_test))

# cvs_all.append(np.mean(scores))



print("SVC POLY Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))

print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))

print("Execution time: {0:.5} seconds \n".format(end-start))

print(classification_report(y_test, prediction))
scores
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest( k=2).fit(X,y)

x_new = selector.transform(X) # not needed to get the score

scores = selector.scores_

scores
X_train[:,0]