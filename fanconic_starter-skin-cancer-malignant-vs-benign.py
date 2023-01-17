from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler



import os



%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from glob import glob

import seaborn as sns

from PIL import Image

print(os.listdir('../input/data/'))
folder_benign_train = '../input/data/train/benign'

folder_malignant_train = '../input/data/train/malignant'



folder_benign_test = '../input/data/test/benign'

folder_malignant_test = '../input/data/test/malignant'



read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))



# Load in training pictures 

ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]

X_benign = np.array(ims_benign, dtype='uint8')

ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in os.listdir(folder_malignant_train)]

X_malignant = np.array(ims_malignant, dtype='uint8')



# Load in testing pictures

ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]

X_benign_test = np.array(ims_benign, dtype='uint8')

ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]

X_malignant_test = np.array(ims_malignant, dtype='uint8')



# Create labels

y_benign = np.zeros(X_benign.shape[0])

y_malignant = np.ones(X_malignant.shape[0])



y_benign_test = np.zeros(X_benign_test.shape[0])

y_malignant_test = np.ones(X_malignant_test.shape[0])





# Merge data 

X_train = np.concatenate((X_benign, X_malignant), axis = 0)

y_train = np.concatenate((y_benign, y_malignant), axis = 0)



X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)

y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)



# Shuffle data

s = np.arange(X_train.shape[0])

np.random.shuffle(s)

X_train = X_train[s]

y_train = y_train[s]



s = np.arange(X_test.shape[0])

np.random.shuffle(s)

X_test = X_test[s]

y_test = y_test[s]
# Display first 15 images of moles, and how they are classified

w=40

h=30

fig=plt.figure(figsize=(12, 8))

columns = 5

rows = 3



for i in range(1, columns*rows +1):

    ax = fig.add_subplot(rows, columns, i)

    if y_train[i] == 0:

        ax.title.set_text('Benign')

    else:

        ax.title.set_text('Malignant')

    plt.imshow(X_train[i], interpolation='nearest')

plt.show()



plt.bar(0, y_train[np.where(y_train == 0)].shape[0], label = 'benign')

plt.bar(1, y_train[np.where(y_train == 1)].shape[0], label = 'malignant')

plt.legend()

plt.title("Training Data")

plt.show()



plt.bar(0, y_test[np.where(y_test == 0)].shape[0], label = 'benign')

plt.bar(1, y_test[np.where(y_test == 1)].shape[0], label = 'malignant')

plt.legend()

plt.title("Test Data")

plt.show()

X_train = X_train/255.

X_test = X_test/255.
# support vector machine classifier

from sklearn.svm import SVC



model = SVC()



model.fit(X_train.reshape(X_train.shape[0],-1), y_train)
from sklearn.metrics import accuracy_score



y_pred = model.predict(X_test.reshape(X_test.shape[0],-1))



print(accuracy_score(y_test, y_pred))