# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



import cv2

from mpl_toolkits.axes_grid1 import ImageGrid



import random



from keras.models import Sequential

#from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D

#from keras.datasets import cifar10

from keras.layers import Conv2D, MaxPooling2D, MaxPool2D

from keras.layers import Dense, Dropout, Activation, Flatten



from keras.optimizers import Adam, RMSprop

from keras.initializers import RandomNormal

import matplotlib.pyplot as plt

import seaborn as sns



from keras import optimizers

import gc



# Any results you write to the current directory are saved as output.
prefix = '../input/images/images'

submission = pd.read_csv('../input/sample_submission.csv')

train_solutions = pd.read_csv('../input/train_solutions.csv')

test_files = list(submission['Id'])

all_files = os.listdir(prefix)

train_files = list(set(all_files) - set(test_files))
print("#All files:  ",len(all_files))

print("#Test files:  ",len(test_files))

print("#Train files: ",len(train_files))

print("#Train solution files: ",train_solutions.shape[0])
#train_solutions.head()

train_solutions_validated = train_solutions[train_solutions['Id'].isin(train_files)]

print("#Validated train solutions:", train_solutions_validated.shape[0]) 

print("one file is missing")
import sys

nsamples = len(train_files)

trainall_x = np.array([None] * nsamples)



for j,i in enumerate(train_solutions_validated['Id'][:nsamples]):

    trainall_x[j] = plt.imread(prefix + '/' + i)

    if not j % 50:

        sys.stdout.write('.')

    if not j % 1000:

        sys.stdout.write(str(j))

        



trainall_y = np.array(train_solutions_validated['Category'][:nsamples]) #np.concatenate([np.zeros(nsamples),np.ones(nsamples)])

classes = ['cat','dog']
len(test_files)
import sys

wrong_file='edb274d8c594429b8ccc684d79d771f6.jpg'

test_files.remove(wrong_file)

nsamples_test = len(test_files)

df_test = np.array([None] * nsamples_test)



for j,i in enumerate(test_files):

    df_test[j] = plt.imread(prefix + '/' + i)

    if not j % 50:

        sys.stdout.write('.')

        

test_x = df_test #np.concatenate([df_train_dog,df_train_cat])
plt.imshow(trainall_x[0])
fig = plt.figure(1, (15., 4.))

grid = ImageGrid(fig, 111, nrows_ncols=(3, 12), axes_pad=0.4)



for i in range(36):

    grid[i].imshow(trainall_x[i])

    grid[i].set_title(classes[trainall_y[i]])

    grid[i].axis('off')





#plt.imshow(train_x[1], title=classes[train_y[1]])
size = (100, 100)

num_channels = 3
j = 0

for i in trainall_x:

    trainall_x[j] = cv2.resize(i, size)

    #train_x[j] = train_x[j].reshape(1, size[0], size[1], num_channels)

    j += 1

    

trainall_x/= 255

trainall_x = np.stack(trainall_x)
j = 0

for i in test_x:

    test_x[j] = cv2.resize(i, size)

    j += 1

    

test_x/= 255

test_x = np.stack(test_x)
fig = plt.figure(1, (15., 4.))

grid = ImageGrid(fig, 111, nrows_ncols=(3, 12), axes_pad=0.4)



for i in range(36):

    grid[i].imshow(trainall_x[i])

    grid[i].set_title(classes[trainall_y[i]])

    grid[i].axis('off')
plt.imshow(trainall_x[4])
cnn_model = Sequential()

cnn_model.add(Conv2D(32, (3, 3), activation='relu',input_shape=trainall_x.shape[1:]))

cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Conv2D(64, (3, 3), activation='relu'))

cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Conv2D(128, (3, 3), activation='relu'))

cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Conv2D(128, (3, 3), activation='relu'))

cnn_model.add(MaxPooling2D((2, 2)))

cnn_model.add(Flatten())

cnn_model.add(Dropout(0.5))  #Dropout for regularization

cnn_model.add(Dense(512, activation='relu'))

cnn_model.add(Dense(1, activation='sigmoid'))
cnn_model.summary()
#cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])



cnn_model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
import numpy as np

from sklearn.model_selection import KFold



kf = KFold(n_splits=5, shuffle=False)

for train_index, valid_index in kf.split(trainall_x):

    print("TRAIN:", len(train_index), "VALIDATION:", len(valid_index))

    train_x, valid_x = trainall_x[train_index], trainall_x[valid_index]

    train_y, valid_y = trainall_y[train_index], trainall_y[valid_index]

    

#using just the last
cnn_history = cnn_model.fit(train_x, train_y, batch_size=100,

                            epochs=50, validation_data=(valid_x, valid_y), shuffle=True)
plt.rcParams['figure.figsize'] = (12, 6)

plt.plot(cnn_history.history['acc'], label='CNN train acc', lw=5, c='b')

plt.plot(cnn_history.history['val_acc'], label='CNN validation acc', lw=5, c='b', ls='--')

plt.xlabel('#epochs', fontsize=15)

plt.ylabel('accuracy', fontsize=15)

plt.ylim(0.3, 1)

plt.legend(fontsize=15)

plt.show()
pred_test_y = (1 * (cnn_model.predict(test_x).flat>0.5 )).astype(int)
print(str(classes[pred_test_y[1996]]))

plt.imshow(test_x[1996])
res=pd.DataFrame({'Id':test_files, 'CategoryPred':pred_test_y})

submission = pd.read_csv('../input/sample_submission.csv')

submission = submission.merge(res, on='Id',how='left').fillna(0)

submission['Category'] = submission['CategoryPred'].astype(int)

submission = submission[['Id','Category']]
#https://www.kaggle.com/dansbecker/submitting-from-a-kernel

submission.to_csv('submission.csv', index=False)