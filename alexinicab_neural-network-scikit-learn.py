# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')  

print(train.shape)

train.head()
test= pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print(test.shape)

test.head()
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

Y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')
import matplotlib.pyplot as plt

%matplotlib inline



#Convert train datset to (num_images, img_rows, img_cols) format 

X_train = X_train.reshape(X_train.shape[0], 28, 28)



for i in range(0, 3):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(Y_train[i]);

    

X_train = X_train.reshape(X_train.shape[0], -1) # Reflattens back the dataset
x_train = X_train[0:38000]

y_train = Y_train[0:38000]

x_dev = X_train[38000:42000]

y_dev = Y_train[38000:42000]
print(x_train.shape)

print(y_train.shape)

print(x_dev.shape)

print(y_dev.shape)
from sklearn.neural_network import MLPClassifier

import time



clf = MLPClassifier(solver='adam', alpha=0.5, hidden_layer_sizes=(400, 100, 50), random_state=1)



# Training

tic = time.time()

clf.fit(x_train, y_train)

toc = time.time()

print("The total training time is", toc-tic, "seconds")
from sklearn import  metrics



#accuracy and confusion matrix on the training set set

predicted_train = clf.predict(x_train)



print("Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_train, predicted_train)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_train, predicted_train))
#accuracy and confusion matrix on the dev set

predicted_dev = clf.predict(x_dev)



print("Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_dev, predicted_dev)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_dev, predicted_dev))
predictions = clf.predict(X_test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("submissions.csv", index=False, header=True)