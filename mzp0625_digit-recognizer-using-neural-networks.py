# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#%% Data extraction

train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

Y = train.label

X = train.drop('label', axis = 1)

X_test = test
#%% Data standardization

X = X/255

X_test = X_test/255
#%% Data exploration

import matplotlib.pyplot as plt

import seaborn as sns

fig, axs = plt.subplots()

g = sns.countplot(Y)
#%% extract first 10 occurances of each digit for visualization

from collections import defaultdict

occurances = defaultdict(list)

for i in range(10):

    for digit in range(10):

        occurances[digit].append(Y[Y==digit].index[i])



fig, axes = plt.subplots(10,10, sharex = True, sharey = True, figsize = (10,12))

axes = axes.flatten()



for digit in occurances:

    for i in range(len(occurances[digit])):

        image = X.values[occurances[digit][i]].reshape(28,28)

        axes[digit*10 + i].imshow(image, cmap = 'gray')

        axes[digit*10 + i].axis('off')

        axes[digit*10 + i].set_title(digit)

plt.tight_layout()
#%% Simple Logistic Regression

from sklearn.linear_model import LogisticRegression

LR_model = LogisticRegression(max_iter = 1000, random_state = 0, solver='lbfgs', multi_class = 'multinomial').fit(X,Y)

prediction = LR_model.predict(X_test)



#%% Print result to CSV

prediction = pd.DataFrame(prediction, columns = ['Label'])

prediction.index += 1

prediction.to_csv(index_label = 'ImageId',path_or_buf = 'LR_model.csv')
#%% Simple Logistic Regression

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

MLP_model = MLPClassifier(random_state = 0)

param_grid = {'hidden_layer_sizes':[(100,),(350,150),(250,150,50)]}

bestModel = GridSearchCV(MLP_model, param_grid, verbose = False, cv = 2).fit(X,Y)

prediction = bestModel.predict(X_test)

#%% Print result to CSV

prediction = pd.DataFrame(prediction, columns = ['Label'])

prediction.index += 1

prediction.to_csv(index_label = 'ImageId',path_or_buf = 'MLP_model_CV.csv')

#%% re-extract data

train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

Y = train.label

X = train.drop('label', axis = 1)

X_test = test



#%% Data standardization

X = X/255

X_test = X_test/255
#%% Split data in to training and validation set

split = 35700

X_train = X[:split]

Y_train = Y[:split]



X_val = X[split:]

Y_val = Y[split:]



#%% categorize data

from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train,10)

Y_val = to_categorical(Y_val,10)

#%% reshape input feature dimension from split X 784 to split X 28 X 28

img_rows, img_cols = 28, 28

input_shape = (28,28,1) # the input shape is 28x28x1 because the pixels are BW

X_train = X_train.values.reshape(split, img_rows, img_cols,1)

X_val = X_val.values.reshape(len(X)-split, img_rows, img_cols,1)
#%% Create Keras Sequential model

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten



CNN_model = Sequential()

batchsize = int(split/20)



CNN_model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = input_shape))

CNN_model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))

CNN_model.add(MaxPooling2D(pool_size = (2,2)))

CNN_model.add(Dropout(0.25))

CNN_model.add(Flatten())

CNN_model.add(Dense(128, activation = 'relu'))

CNN_model.add(Dropout(0.33))

CNN_model.add(Dense(10, activation = 'softmax'))



CNN_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history_CNN = CNN_model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs = 20, batch_size = batchsize)

#%% Prediction of training set

X = X.values.reshape(len(X), img_rows, img_cols,1)

train_pred = CNN_model.predict_classes(X)

sum(train_pred != Y.values)



digits = []

i = 0

for i in range(len(Y.values)):

    if Y.values[i] != train_pred[i]:

        digits.append(i)



fig, axes = plt.subplots(1, 10, sharex = True, sharey = True, figsize = (10,2))

axes = axes.flatten()

for i in range(10):

    image = X[digits[i]].reshape(img_rows, img_cols)

    axes[i].imshow(image, cmap = 'gray')

    axes[i].axis('off')

    axes[i].set_title(str(train_pred[digits[i]]) + ' ('+str(Y[digits[i]]) +')')

plt.tight_layout()