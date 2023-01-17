!pip uninstall --y kaggle

!pip install --upgrade pip

!pip install kaggle==1.5.6
!chmod 600 ../input/kaggle/kaggle.json



!kaggle competitions download -c logistic-classification-diabetes-knn
import numpy as np

import pandas as pd

import seaborn as sns

import random 

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
train = pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv',header=None,skiprows=1)

test = pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv',header=None,skiprows=1)

train = train.dropna()

train = train.drop(5,axis=1)

train = train.drop(4,axis=1)

X = train.iloc[:,1:-1]

y = train.iloc[:,-1:]



print(X.head(20))

print(X.describe())





print(X.shape)

print(y.shape)



test_x = test.drop(5,axis=1)

test_x = test_x.drop(4,axis=1)

test_x = test_x.iloc[:,1:-1]



print(test_x.head(20))

from sklearn.preprocessing import LabelEncoder

import numpy as np

classic = LabelEncoder()

y = classic.fit_transform(y.values)

print('diabetes labels :',np.unique(y))
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.4, random_state=42, stratify=y, shuffle=True)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors=5,p=2)

#5개의 인접한 이웃, 거리측정기준 : 유클리드 

#knn.fit

#knn.fit(X_train,y_train)

from sklearn.neighbors import KNeighborsClassifier



#Setup arrays to store training and test accuracies

neighbors = np.arange(1,100)

train_accuracy =np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i,k in enumerate(neighbors):

    #Setup a knn classifier with k neighbors

    knn = KNeighborsClassifier(n_neighbors=k)

    

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

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train,y_train)
y_train_pred = knn.predict(X_train)

y_test_pred = knn.predict(X_test) #모델을 적용한 test data dml y값 예측치 

print('Misclassified training samples : %d' %(y_train!=y_train_pred).sum())

# 오분류 데이터 갯수 확인 

print('Misclassified test samples : %d' %(y_test!=y_test_pred).sum())
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_test_pred))
from sklearn.metrics import confusion_matrix

y_pred = knn.predict(X_test)

confusion_matrix(y_test,y_pred)



#True negative = 38

#False positive = 35

#True postive = 113

#Fasle negative = 27
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
y_pred_proba = knn.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Knn(n_neighbors=7) ROC curve')

plt.show()

#Area under ROC curve

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_pred_proba)
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors':np.arange(1,100)}

knn = KNeighborsClassifier()

knn_cv= GridSearchCV(knn,param_grid,cv=5)

knn_cv.fit(X,y)
knn_cv.best_score_
knn_cv.best_params_
knn = KNeighborsClassifier(n_neighbors=14)

knn.fit(X_train,y_train)
y_train_pred = knn.predict(X_train)

#y_test_pred = knn.predict(test_x)

y_test_pred = knn.predict(X_test) #모델을 적용한 test data dml y값 예측치 

print('Misclassified training samples : %d' %(y_train!=y_train_pred).sum())

# 오분류 데이터 갯수 확인 

print('Misclassified test samples : %d' %(y_test!=y_test_pred).sum())
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_test_pred))
y_pred = knn.predict(test_x)

test_x.shape
predict = np.array(y_pred[:50]).reshape(-1,1).astype('int')

id = np.array([i for i in range(len(predict))]).reshape(-1,1)

result = np.hstack([id,predict])

df = pd.DataFrame(result,columns=["ID","Label"])

df.to_csv("submission_form.csv",index=False,header=True)
predict
#!kaggle competitions submit -c logistic-classification-diabetes-knn -f submission_form.csv -m "14010974_이기택"