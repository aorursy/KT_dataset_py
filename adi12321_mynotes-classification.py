#Importing libraries

import os

import numpy as np

import pandas as pd

#from sklearn.datasets import fetch_mldata, fetch_openml  # to get data for practice.
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Load data

train=pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')

test=pd.read_csv('/kaggle/input/mnist-in-csv/mnist_test.csv')

train.head(3)
test.head(3)
t= train.isnull().any()

count=0

for i in t:

    if i ==False:

        count+=1

print(count)



#train column count

print(len(train.columns))
#row count

len(train.index)
# make X and Y

X_train = train.iloc[:, train.columns != 'label'] #all columns except label

y_train = train.iloc[:, train.columns == 'label']  #only label column



X_test = test.iloc[:, test.columns !='label']

y_test = test.iloc[:, test.columns == 'label']



X_train.head(2)
y_train.head(2)
%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



sample_digit=X_train.iloc[0] #get any row

sample_digit_image = sample_digit.values.reshape(28,28) #reshape the values



plt.imshow(sample_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')

plt.axis('off')

plt.show()
#label for the item is

y_train.iloc[0]
#shuffling the dataset so that all kfolds get a good combination of digits

import numpy as np



shuffling_index = np.random.permutation(60000) #returns a randomly permuted sequence or permuted range

print (shuffling_index)

X_train,y_train=X_train.iloc[shuffling_index], y_train.iloc[shuffling_index]
#now do training...

# Train for 1 digit ... for 'sample_digit' above or '5'



y_train_5 = (y_train==5) #true fro all 5s and false for other digits

y_test_5 = (y_test==5)



#training with Stochastic Gradient Descent(SGD)

from sklearn.linear_model import SGDClassifier



sgd=SGDClassifier(random_state=1)

sgd.fit(X_train,y_train_5)



sgd.predict([sample_digit])
#Cross validation

from sklearn.model_selection import cross_val_score

cross_val_score(sgd,X_train, y_train_5, cv=3, scoring='accuracy')
#Confusion matrix



from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd,X_train, y_train_5, cv=3)



from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)



#Precision and Recall



from sklearn.metrics import precision_score, recall_score

print(precision_score(y_train_5, y_train_pred))

print(recall_score(y_train_5, y_train_pred))



from sklearn.metrics import f1_score

print(f1_score(y_train_5, y_train_pred))
