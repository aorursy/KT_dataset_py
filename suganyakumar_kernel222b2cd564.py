import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
dataset=pd.read_csv('/kaggle/input/plant Dataset.csv')
dataset
dataset.iloc[:,:]
dataset.shape
x= dataset.iloc[:,0:192].values
x
y= dataset.iloc[:,-1].values
y
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x = sc_x.fit_transform(x)
x
# applying label Encoder convert string into float



from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()

x[:,0] = labelencoder_x.fit_transform(x[:,0])
x
#K-Nearest Neighbour

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x = sc_x.fit_transform(x)
x
from sklearn.model_selection import train_test_split  

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20, random_state=0) 
#Logistic Regression



from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)
y_pred = logmodel.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
Accuracy_logmodel=logmodel.score(x_test,y_test)
print('Accuracy Score',Accuracy_logmodel)
#K-Nearest Neighbor Classifier

from sklearn.neighbors import KNeighborsClassifier

Classifier_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=1)

Classifier_knn.fit(x_train,y_train)
y_pred = Classifier_knn.predict(x_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
Accuracy_Classifier_knn=Classifier_knn.score(x_test,y_test)
print(Accuracy_Classifier_knn)
#Naive Baiyes



from sklearn.naive_bayes import GaussianNB

GNB=GaussianNB()

GNB.fit(x_train,y_train)
y_pred = GNB.predict(x_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
Accuracy_GNB=GNB.score(x_test,y_test)
print(Accuracy_GNB)
#Support Vector Machine

from sklearn.svm import SVC

SVM=SVC(kernel = 'sigmoid')

SVM.fit(x_train,y_train)
y_pred = SVM.predict(x_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
Accuracy_SVM=SVM.score(x_test,y_test)
print(Accuracy_SVM)
#Support Vector Machine

from sklearn.svm import SVC

SVM=SVC(kernel = 'linear')

SVM.fit(x_train,y_train)
y_pred = SVM.predict(x_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
Accuracy_SVM=SVM.score(x_test,y_test)
print(Accuracy_SVM)
#Support Vector Machine

from sklearn.svm import SVC

SVM=SVC(kernel = 'rbf')

SVM.fit(x_train,y_train)
y_pred = SVM.predict(x_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
Accuracy_SVM=SVM.score(x_test,y_test)
print(Accuracy_SVM)
#Support Vector Machine

from sklearn.svm import SVC

SVM=SVC(kernel = 'poly')

SVM.fit(x_train,y_train)
y_pred = SVM.predict(x_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
Accuracy_SVM=SVM.score(x_test,y_test)
print(Accuracy_SVM)
#Visualizing the overall accuracies of performed models through graph

Supervised_learning_Models=["Log Reg", "KNN", "Naive Bayes","SVM"]

Overall_Accuracy_Score=[Accuracy_logmodel,Accuracy_Classifier_knn,Accuracy_GNB,Accuracy_SVM]

sns.barplot(x=Supervised_learning_Models,y=Overall_Accuracy_Score)

plt.xlabel("Supervised Models")

plt.ylabel("Accuracy")

plt.title("Overall Accuracy Score Graph")

plt.show()