!pip install autokeras
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

%matplotlib inline

from autokeras.image.image_supervised import ImageClassifier

np.random.seed(2)



import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
print (train_data.shape)

train_data.head()
print (test_data.shape)

test_data.head()
X_test_id = test_data['id']

train_data['label'].value_counts()
y_train = train_data['label']

X_train = train_data.drop(['label'], axis = 1)

X_test = test_data.drop(['id'], axis = 1)
# Regularization of values

X_train = X_train/255.0

X_test = X_test/255.0



# Reshaping for an Image rather than line data.

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)



#y_train = to_categorical(y_train, num_classes = 10)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
clf = ImageClassifier(verbose=True)

clf.fit(X_train, y_train, time_limit=3 * 60 * 60)

clf.final_fit(X_train, y_val, X_val, y_val, retrain=True)

Y = clf.evaluate(X_val, y_val)

print(Y)
preds = clf.predict(X_test)

preds = np.argmax(preds,axis=1,out=None)
submission = pd.DataFrame({

    'id' : X_test_id,

    'label' : preds

})
submission.to_csv('submission.csv',index=False)