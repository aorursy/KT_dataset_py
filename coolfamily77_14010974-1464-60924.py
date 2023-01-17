!pip uninstall --y kaggle

!pip install --upgrade pip

!pip install kaggle==1.5.6


!chmod 600 ../input/kaggle/kaggle.json
!kaggle competitions download -c mlregression-cabbage-price
!unzip mlregression-cabbage-price.zip
import numpy as np

import pandas as pd

import seaborn as sns

import random 

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
train = pd.read_csv('../input/mlregression-cabbage-price/train_cabbage_price.csv',header=None,skiprows=1)

test = pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv',header=None,skiprows=1)

X = train.iloc[:,1:-2]

y = train.iloc[:,-1:]



print(X.head(20))

print(X.describe())

print(X.info(verbose=True))



print(X.shape)

print(y.shape)



test_x = test.iloc[:,1:-1]



print(test_x.head(20))
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.4, random_state=42, shuffle=True)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors=5,p=2)

#5개의 인접한 이웃, 거리측정기준 : 유클리드 

#knn.fit

#knn.fit(X_train,y_train)

from sklearn.neighbors import KNeighborsRegressor





#Setup arrays to store training and test accuracies

neighbors = np.arange(1,20)

train_accuracy =np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i,k in enumerate(neighbors):

    #Setup a knn classifier with k neighbors

    knn = KNeighborsRegressor(n_neighbors=k,weights='distance')

    

    #Fit the model

    knn.fit(X_train, y_train)

    

    #Compute accuracy on the training set

    train_accuracy[i] = knn.score(X_train, y_train)

    

    #Compute accuracy on the test set

    test_accuracy[i] = knn.score(X_test, y_test) 
#Generate plot

plt.title('k-NN Varying number of neighbors')

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')

plt.plot(neighbors, train_accuracy, label='Training accuracy')

plt.legend()

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.show()

knn = KNeighborsRegressor(n_neighbors=17,weights='distance')

knn.fit(X_train,y_train)
y_train_pred = knn.predict(X_train)

y_test_pred = knn.predict(X_test) #모델을 적용한 test data dml y값 예측치 

print('MisS training samples : %d' %(y_train!=y_train_pred).sum())

# 오분류 데이터 갯수 확인 

print('MisS test samples : %d' %(y_test!=y_test_pred).sum())
y_pred = knn.predict(test_x)

y_pred
predict = np.array(y_pred).reshape(-1,1).astype('int')

id = np.array([i for i in range(len(predict))]).reshape(-1,1)

result = np.hstack([id,predict])

df = pd.DataFrame(result,columns=["ID","Expected"])

df.to_csv("submission_form.csv",index=False,header=True)
predict
#!kaggle competitions submit -c mlregression-cabbage-price -f submission_form.csv -m "14010974_이기택"