import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

np.random.seed(1)

from keras.models import Sequential

from keras.layers import Dense
train = pd.read_csv('../input/train.csv')

X = train.loc[:,'pixel0':'pixel783']

Y = train.loc[:,'label']

X_test = pd.read_csv('../input/test.csv')



for n in range(1,10):

    plt.subplot(1,10,n)

    plt.imshow(X.iloc[n].values.reshape((28,28)),cmap='gray')

    plt.title(Y.iloc[n])
# Create training data set

X_train = X[:40000]

Y_train = Y[:40000]

Y_train = pd.get_dummies(Y_train)



# Create cross-validation set

X_dev = X[40000:42000]

Y_dev = Y[40000:42000]

Y_dev = pd.get_dummies(Y_dev)



print ("number of training examples = " + str(X_train.shape[0]))

print ("number of cross-validation examples = " + str(X_dev.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_dev shape: " + str(X_dev.shape))

print ("Y_dev shape: " + str(Y_dev.shape))
model = Sequential()

model.add(Dense(32, activation='relu', input_dim=784))

model.add(Dense(16, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train.values, Y_train.values, epochs=20, batch_size=64,verbose=2,

          validation_data=(X_dev.values, Y_dev.values))
predictions = model.predict_classes(X_test.values, verbose=0)

predictions_df = pd.DataFrame (predictions,columns = ['Label'])

predictions_df['ImageID'] = predictions_df.index + 1

submission_df = predictions_df[predictions_df.columns[::-1]]

submission_df.to_csv("submission.csv", index=False, header=True)

submission_df.head()