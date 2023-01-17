import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



import keras

from keras.models import Sequential

from keras.layers import *

import keras.backend as K
test = pd.read_csv("../input/forest-cover-type-prediction/test.csv")

train = pd.read_csv("../input/forest-cover-type-prediction/train.csv")
train.head()
X_train_full = train.drop(['Id', 'Cover_Type'], axis=1)

y_train_full = train.Cover_Type - 1

X_test = test.drop('Id', axis=1)

test_id = test.Id



print(X_train_full.shape)

print(X_test.shape)
print(list(zip(range(0,56), X_train_full.columns)))
scaler = MinMaxScaler()

Xs_train_full = scaler.fit_transform(X_train_full)

Xs_test = scaler.transform(X_test)
Xs_train, Xs_valid, y_train, y_valid = train_test_split(Xs_train_full, y_train_full, test_size=0.2, random_state=1, stratify=y_train_full)

print(Xs_train.shape)

print(Xs_valid.shape)
temp = LogisticRegression(max_iter=10000)

temp.fit(Xs_train, y_train)

temp.score(Xs_train, y_train)
np.random.seed(1)



model = Sequential()

model.add(Dense(512, input_shape=(54,), activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(7, activation='softmax'))

model.summary()
opt = keras.optimizers.Adam(lr=0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

h1 = model.fit(Xs_train, y_train, batch_size=20000, epochs=500, 

               validation_data=(Xs_valid, y_valid), verbose=2)
st = 300

epochs = range(st+1, 501)



plt.figure(figsize=[12,6])

plt.subplot(1,2,1)

plt.plot(epochs, h1.history['accuracy'][st:], label='Training Accuracy')

plt.plot(epochs, h1.history['val_accuracy'][st:], label='Validation Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.subplot(1,2,2)

plt.plot(epochs, h1.history['loss'][st:], label='Training Loss')

plt.plot(epochs, h1.history['val_loss'][st:], label='Validation Loss')

plt.xlabel('Epoch')

plt.legend()



plt.show()
K.set_value(model.optimizer.lr, 0.0001)



h2 = model.fit(Xs_train, y_train, batch_size=20000, epochs=500, 

               validation_data=(Xs_valid, y_valid), verbose=2)
st = 0

epochs = range(st+1, 501)



plt.figure(figsize=[12,6])

plt.subplot(1,2,1)

plt.plot(epochs, h2.history['accuracy'][st:], label='Training Accuracy')

plt.plot(epochs, h2.history['val_accuracy'][st:], label='Validation Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.subplot(1,2,2)

plt.plot(epochs, h2.history['loss'][st:], label='Training Loss')

plt.plot(epochs, h2.history['val_loss'][st:], label='Validation Loss')

plt.xlabel('Epoch')

plt.legend()



plt.show()
K.set_value(model.optimizer.lr, 0.00001)



h3 = model.fit(Xs_train, y_train, batch_size=20000, epochs=500, 

               validation_data=(Xs_valid, y_valid), verbose=2)
st = 0

epochs = range(st+1, 501)



plt.figure(figsize=[12,6])

plt.subplot(1,2,1)

plt.plot(epochs, h3.history['accuracy'][st:], label='Training Accuracy')

plt.plot(epochs, h3.history['val_accuracy'][st:], label='Validation Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.subplot(1,2,2)

plt.plot(epochs, h3.history['loss'][st:], label='Training Loss')

plt.plot(epochs, h3.history['val_loss'][st:], label='Validation Loss')

plt.xlabel('Epoch')

plt.legend()



plt.show()
test_pred = model.predict_classes(Xs_test)
test_pred = test_pred + 1
for i in range(1,8):

    print(list(test_pred).count(i))

submission = pd.DataFrame({

    'Id':test_id,

    'Cover_Type':test_pred

})

submission.head()
submission.to_csv('my_submission.csv', index=False)