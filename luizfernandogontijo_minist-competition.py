# basics

import numpy as np 

import pandas as pd

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical





# plotting

import seaborn as sns

import matplotlib.pyplot as plt



# modeling

from keras.models import Sequential

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout

from keras.callbacks import History 

from keras.callbacks import EarlyStopping

test = pd.read_csv('../input/digit-recognizer/test.csv')

train = pd.read_csv('../input/digit-recognizer/train.csv')
print(train.shape) # 784 pixels in each image

train.head() 
print(test.shape) # test dataset do not has label column

test.head()
X_train = (train.iloc[:,1:].values).astype('float32') # make a matrix pixel values for each label(number)

y_train = train.iloc[:,0].values.astype('int32') # number of each image - one number per row

X_test = test.values.astype('float32')
print(X_train.shape)

X_train
print(y_train.shape)

y_train
X_test
# convert rows to matrixes for each number

train.iloc[[1]] # each image with 785 pixels - images with 28x28 pixels



X_train = X_train.reshape(X_train.shape[0], 28, 28) 
plt.imshow(X_train[9], cmap=plt.get_cmap('gray')) # display one image

plt.title(y_train[9]) 
# display more imagens

for i in range(0, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i],cmap=plt.get_cmap('gray'))

    plt.title(y_train[i]);
X_train = X_train.reshape(X_train.shape[0], 28*28,) # back to initial matrix

X_test = X_test.reshape(X_test.shape[0], 28*28,)

print(X_train.shape)

print(X_test.shape)
print(X_train.max(axis=1)) # max values in each row 
# normalizing

scale = np.max(X_train)

X_train /= scale

X_test /= scale
X_train.max(axis=1) # values normalized - max values = 1
# take out the mean to decrease the correlation

mean = np.std(X_train)

X_train -= mean

X_test -= mean
input_shape = X_train.shape[1]

classes = y_train
y_train =  to_categorical(y_train) # number of each set of pixels needs to be categorical
model1 = Sequential()

model1.add(Dense(100, activation='relu', input_shape=(input_shape,)))

model1.add(Dense(10, activation='softmax'))

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()
history_model1 = model1.fit(X_train, y_train, nb_epoch=10, batch_size=16,validation_split=0.1)
history_model1.history['loss'][-1]
results = pd.DataFrame(data = {'model': 'model1', 

                               'loss': [history_model1.history['loss'][-1]], 

                               'accuracy': [history_model1.history['accuracy'][-1]], 

                               'val_loss': [history_model1.history['val_loss'][-1]], 

                               'val_accuracy': [history_model1.history['val_accuracy'][-1]]})

results
model2 = Sequential()

model2.add(Dense(100, activation='relu', input_shape=(input_shape,)))

model2.add(Dense(100, activation='relu', input_shape=(input_shape,)))

model2.add(Dense(10, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_model2 = model2.fit(X_train, y_train, nb_epoch=10, batch_size=16,validation_split=0.1)
results = pd.DataFrame(data = {'model': ['model1', 'model2'], 

                               'loss': [history_model1.history['loss'][-1], history_model2.history['loss'][-1]], 

                               'accuracy': [history_model1.history['accuracy'][-1], history_model2.history['accuracy'][-1]], 

                               'val_loss': [history_model1.history['val_loss'][-1], history_model2.history['val_loss'][-1]], 

                               'val_accuracy': [history_model1.history['val_accuracy'][-1], history_model2.history['val_accuracy'][-1]]})

results

model3 = Sequential()

model3.add(Dense(100, activation='relu', input_shape=(input_shape,)))

model3.add(Dense(100, activation='relu', input_shape=(input_shape,)))

model3.add(Dense(100, activation='relu', input_shape=(input_shape,)))

model3.add(Dense(10, activation='softmax'))

model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_model3 = model3.fit(X_train, y_train, nb_epoch=10, batch_size=16,validation_split=0.1)
results = pd.DataFrame(data = {'model': ['model1', 'model2', 'model3'], 

                               'loss': [history_model1.history['loss'][-1], history_model2.history['loss'][-1], history_model3.history['loss'][-1]], 

                               'accuracy': [history_model1.history['accuracy'][-1], history_model2.history['accuracy'][-1], history_model3.history['accuracy'][-1]], 

                               'val_loss': [history_model1.history['val_loss'][-1], history_model2.history['val_loss'][-1], history_model3.history['val_loss'][-1]], 

                               'val_accuracy': [history_model1.history['val_accuracy'][-1], history_model2.history['val_accuracy'][-1], history_model3.history['val_accuracy'][-1]]})

results
model4 = Sequential()

model4.add(Dense(100, activation='relu', input_shape=(input_shape,)))

model4.add(Dense(100, activation='relu', input_shape=(input_shape,)))

model4.add(Dense(100, activation='relu', input_shape=(input_shape,)))

model4.add(Dense(100, activation='relu', input_shape=(input_shape,)))

model4.add(Dense(10, activation='softmax'))

model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_model4 = model4.fit(X_train, y_train, nb_epoch=10, batch_size=16,validation_split=0.1)
results = pd.DataFrame(data = {'model': ['model1', 'model2', 'model3', 'model4'], 

                               'loss': [history_model1.history['loss'][-1], history_model2.history['loss'][-1], history_model3.history['loss'][-1], history_model4.history['loss'][-1]], 

                               'accuracy': [history_model1.history['accuracy'][-1], history_model2.history['accuracy'][-1], history_model3.history['accuracy'][-1], history_model4.history['accuracy'][-1]], 

                               'val_loss': [history_model1.history['val_loss'][-1], history_model2.history['val_loss'][-1], history_model3.history['val_loss'][-1], history_model4.history['val_loss'][-1]], 

                               'val_accuracy': [history_model1.history['val_accuracy'][-1], history_model2.history['val_accuracy'][-1], history_model3.history['val_accuracy'][-1], history_model4.history['val_accuracy'][-1]]})

results
model5 = Sequential()

model5.add(Dense(50, activation='relu', input_shape=(input_shape,)))

model5.add(Dense(50, activation='relu', input_shape=(input_shape,)))

model5.add(Dense(50, activation='relu', input_shape=(input_shape,)))

model5.add(Dense(50, activation='relu', input_shape=(input_shape,)))

model5.add(Dense(10, activation='softmax'))

model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_model5 = model5.fit(X_train, y_train, nb_epoch=10, batch_size=16,validation_split=0.1)
results = pd.DataFrame(data = {'model': ['model1', 'model2', 'model3', 'model4', 'model5'], 

                               'loss': [history_model1.history['loss'][-1], history_model2.history['loss'][-1], history_model3.history['loss'][-1], history_model4.history['loss'][-1], history_model5.history['loss'][-1]], 

                               'accuracy': [history_model1.history['accuracy'][-1], history_model2.history['accuracy'][-1], history_model3.history['accuracy'][-1], history_model4.history['accuracy'][-1], history_model5.history['accuracy'][-1]], 

                               'val_loss': [history_model1.history['val_loss'][-1], history_model2.history['val_loss'][-1], history_model3.history['val_loss'][-1], history_model4.history['val_loss'][-1], history_model5.history['val_loss'][-1]], 

                               'val_accuracy': [history_model1.history['val_accuracy'][-1], history_model2.history['val_accuracy'][-1], history_model3.history['val_accuracy'][-1], history_model4.history['val_accuracy'][-1], history_model5.history['val_accuracy'][-1]]})

results
model6 = Sequential()

model6.add(Dense(50, activation='relu', input_shape=(input_shape,)))

model6.add(Dense(50, activation='relu', input_shape=(input_shape,)))

model6.add(Dense(50, activation='relu', input_shape=(input_shape,)))

model6.add(Dense(10, activation='softmax'))

model6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_model6 = model6.fit(X_train, y_train, nb_epoch=10, batch_size=16,validation_split=0.1)
results = pd.DataFrame(data = {'model': ['model1', 'model2', 'model3', 'model4', 'model5', 'model6'], 

                               'loss': [history_model1.history['loss'][-1], history_model2.history['loss'][-1], history_model3.history['loss'][-1], history_model4.history['loss'][-1], history_model5.history['loss'][-1], history_model6.history['loss'][-1]], 

                               'accuracy': [history_model1.history['accuracy'][-1], history_model2.history['accuracy'][-1], history_model3.history['accuracy'][-1], history_model4.history['accuracy'][-1], history_model5.history['accuracy'][-1], history_model6.history['accuracy'][-1]], 

                               'val_loss': [history_model1.history['val_loss'][-1], history_model2.history['val_loss'][-1], history_model3.history['val_loss'][-1], history_model4.history['val_loss'][-1], history_model5.history['val_loss'][-1], history_model6.history['val_loss'][-1]], 

                               'val_accuracy': [history_model1.history['val_accuracy'][-1], history_model2.history['val_accuracy'][-1], history_model3.history['val_accuracy'][-1], history_model4.history['val_accuracy'][-1], history_model5.history['val_accuracy'][-1], history_model6.history['val_accuracy'][-1]]})

results
plt.plot(history_model1.history['val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Validation score')

plt.show()
preds_model1 = model1.predict_classes(X_test, verbose=0)
y_pred = model1.predict(X_test)

y_predict_classes = model1.predict_classes(X_test)
y_predict_classes
y_test = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

y_test = y_test['Label']

y_test = y_test.astype('int32')
Y_pred = np.argmax(y_pred, 1)

print(Y_pred)
def write_preds(preds, fname):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(y_predict_classes, "to_submit.csv")
pd.read_csv('to_submit.csv')