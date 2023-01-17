# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pandas as pd
DF = pd.read_csv('../input/Churn_Modelling.csv')

DF.head(5)
DF.set_index('RowNumber')
DF.columns
DF =DF.drop(['CustomerId','Surname'],axis=1)
DF['Gender'].value_counts()
DF['gender_Cat']=0

DF.loc[(DF['Gender']=='Female'), 'gender_Cat'] = 1
print(DF['gender_Cat'].value_counts())

DF.ix[1:10,['Gender','gender_Cat']]
from sklearn.preprocessing import LabelEncoder

label_X_country_encoder = LabelEncoder()

DF['Geography_cat'] = label_X_country_encoder.fit_transform(DF['Geography'])

DF[DF['Balance']==0.0].count()
DF['Exited'].value_counts()
X = DF.drop(['Exited','Gender','Geography'],axis=1) # Credit Score through Estimated Salary - features

y = DF['Exited'] # Exited target



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

import keras

import tensorflow as tf

tf.set_random_seed(42)

#Initialize Sequential model

model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Dense(activation = 'relu', input_dim = 11, units=6, kernel_initializer='uniform'))

model.add(tf.keras.layers.Dense(activation = 'relu', units=6, kernel_initializer='uniform')) 

model.add(tf.keras.layers.Dense(activation = 'sigmoid', units=1, kernel_initializer='uniform')) 



from keras import optimizers



sgd = tf.keras.optimizers.SGD(lr=0.03)

#Compile the model

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])



model.summary()



history_sgd=model.fit(X_train, y_train, batch_size=10, epochs=40)
y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)



test_loss, test_acc = model.evaluate(x=X_test,y=y_test.values)

print("Accuracy: ",test_acc)

print("Loss: ",test_loss)
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat)



labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
#Initialize Sequential model

model2 = tf.keras.models.Sequential()

model2.add(tf.keras.layers.Dense(activation = 'relu', input_dim = 11, units=8))

model2.add(tf.keras.layers.Dense(activation = 'relu', units=8)) 

model2.add(tf.keras.layers.Dense(activation = 'sigmoid', units=1)) 

model2.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

model2.summary()

history_adam=model2.fit(X_train, y_train, batch_size=10, epochs=40)



#history_adam=model2.fit(X_train, y_train, batch_size=10, epochs=40)
y_pred = model2.predict(X_test)

y_pred = (y_pred > 0.5)

test_loss, test_acc = model2.evaluate(x=X_test,y=y_test)

print("Accuracy: ",test_acc)

print("Loss: ",test_loss)



conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat)
from imblearn.over_sampling import SMOTE

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

ratio='minority'

#sm = SMOTE(random_state=2)

sm = SMOTE(ratio='minority')

#X_sm, y_sm = smote.fit_sample(X, y.ravel())

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)



print('After OverSampling, the shape of X_train: {}'.format(X_train_res.shape))

print('After OverSampling, the shape of y_train: {} \n'.format(y_train_res.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

#Normalize the data



X_train_res = sc.fit_transform(X_train_res)
#Initialize Sequential model

model3 = tf.keras.models.Sequential()

model3.add(tf.keras.layers.Dense(activation = 'relu', input_dim = 11, units=6, kernel_initializer='uniform'))

model.add(tf.keras.layers.Dropout(0.6))

model3.add(tf.keras.layers.Dense(activation = 'relu', units=6, kernel_initializer='uniform')) 

model3.add(tf.keras.layers.Dense(activation = 'sigmoid', units=1, kernel_initializer='uniform')) 

sgd = tf.keras.optimizers.SGD(lr=0.01)

#Compile the model

model3.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])



#adam=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)



model3.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics=['accuracy'])

model3.summary()



history_smote_sgd=model3.fit(X_train_res, y_train_res, epochs=40)
y_pred = model3.predict(X_test)

y_pred = (y_pred > 0.5)

test_loss, test_acc = model3.evaluate(x=X_test,y=y_test)

print("Accuracy: ",test_acc)

print("Loss: ",test_loss)



conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat)
from matplotlib import pyplot

# plot metrics

pyplot.plot(history_sgd.history['acc'],label='sgd')

pyplot.plot(history_adam.history['acc'],label='adam')

pyplot.plot(history_smote_sgd.history['acc'],label='smote_adam')

#pyplot.legend([line1, line2, line3], ['sgd', 'adam', 'smote_adam'])

pyplot.legend()



pyplot.show()