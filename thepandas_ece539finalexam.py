import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import SGD,Adam
from sklearn.metrics import confusion_matrix
import keras

f = np.loadtxt('../input/ctrain.txt')
y_train = f[:,4].astype(int)
x_train = f[:,0:4]

f1 = np.loadtxt('../input/ctest.txt')
x_test = f1[:,0:4]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')
model = Sequential()

model.add(Dense(30,input_shape=(4,),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15,activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])

print(model.summary())
history = model.fit(x_train, y_train, validation_split = 0.1, epochs = 300,batch_size = 10)
model.predict_classes(x_test)
d = np.loadtxt('../input/p1train.txt')
y_train1 = d[:,8].astype(int)
x_train1 = d[:,0:8]

d1 = np.loadtxt('../input/p1test.txt')
x_test1 = d1[:,0:8]
y_test1 = d1[:,8].astype(int)
y_test1
sc = StandardScaler()
x_train1 = sc.fit_transform(x_train1)
x_test1 = sc.transform(x_test1)
w_opt = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_train1),x_train1)),np.transpose(x_train1)),y_train1)
print(w_opt)
pred_ytest = np.sign(np.dot(w_opt,np.transpose(x_Test1)))
err = np.sum(np.absolute(pred_ytest-y_Test1))/2
err.astype(int)
slp = Sequential()

slp.add(Dense(1,input_shape=(8,),activation='linear',kernel_initializer="zero"))


sgd = SGD(lr=0.01) #decay=1e-6, momentum=0.9, nesterov=True)
slp.compile(loss='mse', optimizer=sgd,metrics=['accuracy',])

#model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])

slp.summary()
history1 = slp.fit(x_train1, y_train1, epochs = 100,batch_size=5,validation_split=0.1)
y_pred1 = slp.predict(x_test1)
y_pred1 = np.sign(y_pred1).astype(int)
np.sum(np.absolute(y_pred1-np.reshape(y_test1,(60,1))))
slp.get_weights()
print(w_opt)
y_pred_training = np.sign(slp.predict(x_train1))
y_pred_testing = np.sign(slp.predict(x_test1))
from sklearn.metrics import confusion_matrix as cm
cm(y_test1,y_pred_testing)
cm(y_train1,y_pred_training)
from sklearn.svm import LinearSVC
clf = LinearSVC(random_state = 0, tol = 1e-5)
clf.fit(x_train1,y_train1)
print(clf.coef_)
y_pred_svm_test = clf.predict(x_test1)
y_pred_svm_train = clf.predict(x_train1)
cm(y_test1,y_pred_svm_test)
cm(y_train1,y_pred_svm_train)
