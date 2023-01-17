import numpy as np

import pandas as pd



dataset = pd.read_csv('../input/heart-disease-uci/heart.csv')

#dataset.head(5)

#dataset.info()



chest_pain=pd.get_dummies(dataset['cp'], prefix='cp', drop_first=True)

dataset=pd.concat([dataset, chest_pain], axis=1)

dataset.drop(['cp'], axis=1, inplace=True)

sp=pd.get_dummies(dataset['slope'], prefix='slope')

th=pd.get_dummies(dataset['thal'], prefix='thal')

rest_ecg=pd.get_dummies(dataset['restecg'], prefix='restecg')

frames=[dataset, sp, th, rest_ecg]

dataset=pd.concat(frames, axis=1)

dataset.drop(['slope','thal','restecg'], axis=1, inplace=True)



X = dataset.drop(['target'], axis = 1)

y = dataset.target.values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.18, random_state=0) 



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

import keras

import warnings



classifier=Sequential()

classifier.add(Dense(output_dim=13, init='uniform', activation='relu', input_dim=22))

classifier.add(Dropout(rate=0.18, noise_shape=None, seed=None))

classifier.add(Dense(output_dim=9, init='uniform', activation='relu'))

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))



classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



classifier.fit(X_train, y_train, batch_size=12, nb_epoch=120)



y_pred = classifier.predict(X_test)



import seaborn as sns

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred.round())



total=sum(sum(cm))



sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])

print('Sensitivity : ', sensitivity )



specificity = cm[1,1]/(cm[1,1]+cm[0,1])

print('Specificity : ', specificity)



from sklearn.metrics import accuracy_score

ac=accuracy_score(y_test, y_pred.round())

print('Accuracy: ',ac)