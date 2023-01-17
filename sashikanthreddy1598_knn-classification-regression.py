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
#import the required packages for solving 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix,mean_squared_error,accuracy_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/Train_sample.csv')

test = pd.read_csv('../input/Test_sample.csv')

train.head()
train.tail()
test.head()
test.tail()
X_train = train.iloc[:,1:]

y_train = train.iloc[:,0]



print(X_train.shape)

print(y_train.shape)
X_test = test.iloc[:,1:]

y_test = test.iloc[:,0]



print(X_test.shape)

print(y_test.shape)
x1 = X_train.iloc[0,:].values.reshape(28,28)

x1[x1> 0] =1

x1 = pd.DataFrame(x1)

x1.to_csv("one.csv")
train_sample = np.random.choice(range(0,X_train.shape[0]),replace=False, size=5)

test_sample = np.random.choice(range(0,X_test.shape[0]),replace = False, size =5)
train_sample
test_sample
plt.figure(figsize=(10,5))

for i,j in enumerate(train_sample):

    plt.subplot(2,5,i+1)

    plt.imshow(X_train.iloc[j,:].values.reshape(28,28))

    plt.title("Digit:" +str(y_train[j]))

    plt.gray()
plt.figure(figsize=(10,5))

for i,j in enumerate(test_sample):

    plt.subplot(2,5,i+1)

    plt.imshow(X_test.iloc[j,:].values.reshape(28,28))

    plt.title("Digit:"+str(y_test[j]))

    plt.gray()
knn_classifier = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='brute')

knn_classifier.fit(X_train, y_train)
pred_train = knn_classifier.predict(X_train)

pred_test = knn_classifier.predict(X_test)
cm_test = confusion_matrix(y_pred=pred_test, y_true=y_test)



print(cm_test)
#Accuracy:

sum(np.diag(cm_test))/np.sum(cm_test)



#np.trace(cm_test)/np.sum(cm_test)
print("Accuracy on train is:" , accuracy_score(y_train,pred_train))

print("Accuracy on test is:", accuracy_score(y_test,pred_test))
misclassified = y_test[pred_test != y_test]
##First 5 misclassified ponts

misclassified.index[:5]
plt.figure(figsize=(10,5))

for i,j in enumerate(misclassified.index[:5]):

    plt.subplot(2,5,i+1)

    plt.imshow(X_test.iloc[j,:].values.reshape(28,28))

    plt.title("Digit:"+str(y_test[j])+" "+"Pred:"+str(pred_test[j]))

    plt.gray()
knn_classifier = KNeighborsClassifier(algorithm= 'brute', weights='distance')
params = {"n_neighbors": [1,3,5],"metric": ["euclidean", "cityblock"]}

#params = {"n_neighbors": [1],"metric": ["euclidean", "cityblock"]}



grid = GridSearchCV(knn_classifier,param_grid=params,scoring="accuracy",cv=10)
grid.fit(X_train, y_train)

print(grid.best_score_)

print(grid.best_params_)
best_knn = grid.best_estimator_

pred_train = best_knn.predict(X_train)

pred_test = best_knn.predict(X_test)

print("Accuracy o train is:", accuracy_score(y_train,pred_train))

print("Accuracy on test is:", accuracy_score(y_test,pred_test))
##Randomly generate some data



data  = pd.DataFrame(np.random.randint(low = 2,high = 100,size = (1000, 4)),

                     columns=["Target","A","B","C"])

data.head()
train_x,test_x,train_y,test_y = train_test_split(data.iloc[:,1:],data.Target,test_size = 0.2)

print(train_x.shape, test_x.shape)
scaler = MinMaxScaler(feature_range=(0,1))



scaler.fit(train_x)
scaled_train_x = pd.DataFrame(scaler.transform(train_x),columns=['A','B','C'])

scaled_test_x = pd.DataFrame(scaler.transform(test_x),columns=["A","B","C"])
knn_regressor = KNeighborsRegressor(n_neighbors=3,algorithm="brute",weights="distance")

knn_regressor.fit(scaled_train_x, train_y)
train_pred = knn_regressor.predict(scaled_train_x)

test_pred = knn_regressor.predict(scaled_test_x)
print(mean_squared_error(train_y,train_pred))

print(mean_squared_error(test_y,test_pred))
knn_regressor = KNeighborsRegressor(algorithm="brute",weights="distance")

params = {"n_neighbors": [1,3,5],"metric": ["euclidean", "cityblock"]}

grid = GridSearchCV(knn_regressor,param_grid=params,scoring="neg_mean_squared_error",cv=5)
grid.fit(scaled_train_x, train_y)

print(grid.best_params_)

print(grid.best_score_)
best_knn = grid.best_estimator_

train_pred = best_knn.predict(scaled_train_x)

test_pred = best_knn.predict(scaled_test_x)
print(mean_squared_error(train_y,train_pred))

print(mean_squared_error(test_y,test_pred))