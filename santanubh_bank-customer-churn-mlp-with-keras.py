# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.metrics import confusion_matrix



# import os

# print(os.listdir("../input"))



#Load Dataset

dataset = pd.read_csv('../input/Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1].values

y = dataset.iloc[:, -1].values
#Lable encoding

categorical_features = [1, 2]

label_encoder_list = []

for feature in categorical_features:

    label_encoder = LabelEncoder()

    X[:, feature] = label_encoder.fit_transform(X[:, feature])

    label_encoder_list.append(label_encoder)



#One Hot Encoding

shift = categorical_features[0]

prev_no_features = X.shape[1]

for feature in categorical_features:

    one_hot_encoder = OneHotEncoder(categorical_features=[shift])

    X = one_hot_encoder.fit_transform(X).toarray()

    X = X[:, 1:]

    shift = X.shape[1] - prev_no_features + 2

    no_features = X.shape[1]
#Feature Scaling

scaler = StandardScaler()

X = scaler.fit_transform(X)



#Spliting dataset into train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("X_train shape - ",X_train.shape)

print("y_train shape - ", y_train.shape)
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



#MLP model

model = Sequential()



model.add(Dense(22, input_shape=(X_train.shape[1],)))

model.add(Dropout(0.2))

model.add(Activation('relu'))



model.add(Dense(16))

model.add(Dropout(0.2))

model.add(Activation('relu'))



model.add(Dense(12))

model.add(Activation('relu'))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.summary()
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=16, epochs=100)
loss, accuracy = model.evaluate(X_test, y_test)

print("Model Accuracy --> ", accuracy)
y_hat = np.rint(model.predict(X_test))

print("Confusion Metrix --> \n", confusion_matrix(y_test, y_hat))