# import libraries

import numpy as np # 

import pandas as pd # 

%matplotlib inline
# load the training data and inspect

digits = pd.read_csv('../input/train.csv')

#digits = pd.read_csv('data/train.csv')

digits.head()
# let's look at the structure of one row (image)

a = digits.iloc[3,1:].values

b = a.reshape((28,28))

print(b)
# plot one of the images

import matplotlib.pyplot as plt

plt.imshow(b)
# split the training data into predictors (X) and targets (y)

X=digits.iloc[:,1:].values

y=digits.iloc[:,0].values



# Scale the data to be between 0 and 1 (* this makes big difference in results!*)

X=np.divide(X,255.)



# create dummie variables for the target

from keras.utils.np_utils import to_categorical

y = to_categorical(y)
# Now import keras libraries and start building model

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping
model = Sequential()

model.add(Dense(100,activation='relu',input_shape=(784,)))

model.add(Dense(15,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# the EarlyStopping stops the model training if it does not improve after 5 epochs

history=model.fit(X,y,validation_split=0.3,callbacks=[EarlyStopping(patience=5)],epochs=50)

# plot loss vs # model epochs trained

plt.plot(history.history['val_loss'])

plt.plot(history.history['loss'])

plt.grid()

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(['valid','train'])
# plot accuracy vs # epochs

plt.plot(history.history['val_acc'])

plt.plot(history.history['acc'])

plt.legend(['valid','train'])

plt.grid()

plt.xlabel('epoch')

plt.ylabel('accuracy')
# predict on test set and write csv file for submission

#test = pd.read_csv('data/test.csv')

test = pd.read_csv('../input/test.csv')

test.head()

preds = model.predict_classes( np.divide(test.values,255.) )

results = pd.DataFrame({'Label':preds})

results.head()

results.index+=1

results = results.reindex(results.index.rename('ImageId'))

results.head()
# results.to_csv('results.csv', header=True)