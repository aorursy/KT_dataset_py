import pandas as pd
import numpy as np
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
%pylab inline
# load training data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# check shape of training data
train.shape # (42000, 785)

# Dividing dataset into training and CV data using (80:20) ratio
num_train = int(0.8 * train.shape[0])
X = (train.iloc[:num_train, 1:]).values # first 80% examples
X_cv = (train.iloc[num_train:, 1:]).values # rest examples

# observing the dataset
print(X.shape)
print(X)
# get labels from 1st column of each row
labels = train.iloc[:, 0].values

# encoding for training data
y = np.zeros((num_train, 10))
for i in range(num_train):
    y[i][labels[i]] = 1;

num_cv = train.shape[0] - num_train
# encoding for cv data
y_cv = np.zeros((num_cv, 10))
for i in range(num_cv):
    y_cv[i][labels[num_train + i]] = 1
    
# training labels
print(y)

print("---------------------------------------------")

# cv labels
print(y_cv)
model = Sequential([
    # input layer
    Dense(32, input_dim=784),
    Activation('sigmoid'),
    Dropout(0.25), # 25% dropout rate to prevent overfitting
    # hidden layer
    Dense(32),
    Activation('sigmoid'),
    Dropout(0.25),
     # hidden layer
    Dense(32),
    Activation('sigmoid'),
    Dropout(0.25),
     # hidden layer
    Dense(32),
    Activation('sigmoid'),
    Dropout(0.25),
    # output layer (10 nodes)
    Dense(10),
    Activation('sigmoid'),
])
# compiling model
model.compile(optimizer = 'adadelta',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
# training the model
hist = model.fit(X, y, epochs = 50, batch_size = 32, verbose = 0)
print('done')
score = model.evaluate(X_cv,y_cv,batch_size = 32, verbose = 0)
print(score)
info = hist.history
plt.plot(info['loss'])
yPred = model.predict_classes(test)
print(yPred)
np.savetxt('submission.csv', np.c_[range(1, len(yPred) + 1), yPred], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
! ls
sample = pd.read_csv('../input/sample_submission.csv')
sample