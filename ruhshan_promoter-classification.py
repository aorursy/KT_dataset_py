# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from Bio import SeqIO
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import math
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
window = 16

def make_eiin_matrix(s):
    """
    This function takes a DNA sequence and makes a eiin matrix .
    """
    length = len(s)
    eiins = {'A':0.1260, 'T':0.1335,'C':0.1340, 'G':0.0806, 'N':0.0}
    # Making empty eiin matrix
    
    eiin_matrix = np.zeros(shape = (window, length))
    
    # Populating first row of matrix with exact eiin values
    for i in range(len(s)):
        eiin_matrix[0][i] = eiins[s[i]]
    
    for i in range(1, window):
        
        for j in range(length):
            consecutives = eiin_matrix[0][j:j+i+1]
            eiin_matrix[i][j] = np.sum(consecutives)
        
    return eiin_matrix
!grep -c ">" ../input/ctv_si/si_total.txt
!grep -c ">" ../input/ctv_si/ctv_total.txt
# Making Stress inducible dataset
rows = 1000
sequence_length = 121
stress_inducibles = np.zeros(shape=(rows,sequence_length * window))
ct = 0

for s in SeqIO.parse('../input/ctv_si/si_total.txt', 'fasta'):
    target_subsequence = str(s.seq)[-sequence_length:]
    eiin_matrix = make_eiin_matrix(target_subsequence.upper())
    stress_inducibles[ct] = eiin_matrix.flatten()
    ct+=1
    if ct ==rows:
        break
# Making constitutive inducible dataset
constitutives = np.zeros(shape=(rows,sequence_length * window))
ct = 0

for s in SeqIO.parse('../input/ctv_si/ctv_total.txt', 'fasta'):
    target_subsequence = str(s.seq)[-sequence_length:]
    eiin_matrix = make_eiin_matrix(target_subsequence.upper())
    constitutives[ct] = eiin_matrix.flatten()
    ct+=1
    if ct ==rows:
        break
# making features and labels

X = np.vstack([stress_inducibles, constitutives])
y = np.append(np.ones(rows) , np.zeros(rows))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

mat_row=mat_col = int(math.sqrt(X.shape[1]))

X_train = X_train.reshape(X_train.shape[0], mat_row, mat_col)
X_test = X_test.reshape(X_test.shape[0], mat_row, mat_col)

num_classes = 2
input_shape = (mat_row, mat_col, 1)
batch_size = 10
epochs = 20

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

import matplotlib.pyplot as plt

#plt.imshow(X_test[4])

fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 9),
                        subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)

for ax, i in zip(axs.flat, range(18)):
    ax.imshow(X_train[i])
    if y_train[i][0]==0.0:
        ax.set_title('si')
    else:
        ax.set_title('co')
y_train[0][0]==0.0
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# model.add(Flatten())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])