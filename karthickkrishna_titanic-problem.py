# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
titatrain = pd.read_csv("../input/train.csv")  #importing the train data
titatest = pd.read_csv("../input/test.csv")   #importing the test data

# Any results you write to the current directory are saved as output.
titatrain.head()   #To view the head of the train data set 
titatest.head()     #To view the head of the test data set
titatrain.info()   #Finding the information(such as number of datas missing) on the training dataset
titatest.info()
titatrain.describe()   #To see the max, min and other details that will help us
titatrain.corr()
print(titatrain.keys())
print(titatest.keys())
plt.figure(figsize=(10,6))
sns.heatmap(pd.isnull(titatrain))   #To see the missing values, the Cabin column has lots of missing values, The Age column might be useful to see what age people survived
sns.distplot(titatrain['Age'].dropna(),bins=50)  # ".dropna()" because, in our dataset there are NAN and we will get a Value Error if dropna is not present
sns.distplot(titatest['Age'].dropna(),bins=50)
sns.boxplot('Pclass','Age',data=titatrain)
PC1 = titatrain[titatrain['Pclass']==1]['Age'].mean()  #Finding the mean age of Pclass 1
PC2 = titatrain[titatrain['Pclass']==2]['Age'].mean()  #Finding the mean age of Pclass 2
PC3 = titatrain[titatrain['Pclass']==3]['Age'].mean()  #Finding the mean age of Pclass 3
def mis(cont):                   #Will be using this function for test dataset too.
    Age = cont[0]
    Pclass = cont[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return PC1
        elif Pclass == 2:
            return PC2
        else:
            return PC3
    else:
        return Age
titatrain['Age'] = titatrain[['Age','Pclass']].apply(mis,axis=1)
titatrain.info()
sns.boxplot('Pclass','Age',data=titatest)
PC1 = titatest[titatest['Pclass']==1]['Age'].mean()  #Finding the mean age of Pclass 1
PC2 = titatest[titatest['Pclass']==2]['Age'].mean()  #Finding the mean age of Pclass 2
PC3 = titatest[titatest['Pclass']==3]['Age'].mean()  #Finding the mean age of Pclass 3
titatest['Age'] = titatest[['Age','Pclass']].apply(mis,axis=1)
titatest.info()
plt.figure(figsize=(13,6))
sns.countplot('Survived',data=titatrain,hue='Sex').margins(x=0)
plt.legend( loc = 'upper right')
plt.figure(figsize=(13,6))
sns.countplot('Survived',hue='Pclass',data=titatrain).margins(x=0)
plt.figure(figsize=(13,6))
sns.countplot('SibSp',data=titatrain).margins(x=0)
sns.distplot(titatrain['Age'],kde=False,bins=20).margins(x=0)
surages = titatrain[titatrain.Survived == 1]["Age"]
notsurages = titatrain[titatrain.Survived == 0]["Age"]
plt.subplot(1, 2, 1)
sns.distplot(surages, kde=False).margins(x=0)
plt.title('Survived')
plt.subplot(1, 2, 2)
sns.distplot(notsurages, kde=False)
plt.title('Not Survived')
plt.subplots_adjust(right=2)
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=titatrain)
sns.pairplot(titatrain,hue = 'Survived')
Sextrain = pd.get_dummies(titatrain['Sex'],drop_first=True)
Embarkedtrain = pd.get_dummies(titatrain['Embarked'],drop_first=True)
Pclasstrain = pd.get_dummies(titatrain['Pclass'],drop_first=True)

Sextest =pd.get_dummies(titatest['Sex'],drop_first=True)
Embarkedtest = pd.get_dummies(titatest['Embarked'],drop_first=True)
Pclasstest = pd.get_dummies(titatest['Pclass'],drop_first=True)
titatrain = pd.concat([titatrain,Sextrain,Embarkedtrain,Pclasstrain],axis=1)
titatest = pd.concat([titatest,Sextest,Embarkedtest,Pclasstest],axis=1)
titatrain.drop(['Embarked','Sex','Pclass','PassengerId','Cabin','Name','Ticket'],axis=1,inplace=True)
titatrain.head()
titatest.drop(['Pclass','Name','Sex','Ticket','Embarked','Cabin'],axis=1,inplace=True)
titatest.head()
X_train = titatrain.drop('Survived',axis=1)  #We define the training label set
y_train = titatrain['Survived']   #We define the training label set
titatest.fillna('0',inplace=True)
X_test = titatest.drop('PassengerId',axis=1)     #We define the testing label set
#we don't have y_test, that is what we're trying to predict with our model
from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size = 0.3,random_state = 0)
from sklearn.preprocessing import StandardScaler

Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.fit_transform(X_test)
X_valid = Sc_X.fit_transform(X_valid)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score #To evaluate the model performance
from sklearn.metrics import classification_report,confusion_matrix
lomo_train = LogisticRegression()
lomo_train.fit(X_train,y_train)
lomo_predictions_train = lomo_train.predict(X_valid)
print(classification_report(y_valid,lomo_predictions_train))
print(confusion_matrix(y_valid,lomo_predictions_train))
acc_lomo = accuracy_score(y_valid, lomo_predictions_train)
acc_lomo
knn_train = KNeighborsClassifier(n_neighbors=1)
knn_train.fit(X_train,y_train)
knn_predictions_train = knn_train.predict(X_valid)
print(classification_report(y_valid,knn_predictions_train))
print(confusion_matrix(y_valid,knn_predictions_train))
acc_knn1 = accuracy_score(y_valid, knn_predictions_train)
acc_knn1
error_rate = []
for i in range(1,50):
    
    knn_train = KNeighborsClassifier(n_neighbors=i)
    knn_train.fit(X_train,y_train)
    knn_predictions_train_i = knn_train.predict(X_valid)
    error_rate.append(np.mean(knn_predictions_train_i != y_valid))
plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn_train = KNeighborsClassifier(n_neighbors=22)
knn_train.fit(X_train,y_train)
knn_predictions_train = knn_train.predict(X_valid)
print(classification_report(y_valid,knn_predictions_train))
print(confusion_matrix(y_valid,knn_predictions_train))
acc_knn = accuracy_score(y_valid, knn_predictions_train)
acc_knn
dtree_train = DecisionTreeClassifier()
dtree_train.fit(X_train,y_train)
dtree_predictions_train = dtree_train.predict(X_valid)
print(classification_report(y_valid,dtree_predictions_train))
print(confusion_matrix(y_valid,dtree_predictions_train))
acc_dtree = accuracy_score(y_valid, dtree_predictions_train)
acc_dtree
rfc_train = RandomForestClassifier(n_estimators=100)
rfc_train.fit(X_train, y_train)
rfc_predictions_train = rfc_train.predict(X_valid)
print(classification_report(y_valid,rfc_predictions_train))
print(confusion_matrix(y_valid,rfc_predictions_train))
acc_rfc = accuracy_score(y_valid, dtree_predictions_train)
acc_rfc
svm_train = SVC()
svm_train.fit(X_train, y_train)
svm_predictions_train = svm_train.predict(X_valid)
print(classification_report(y_valid,svm_predictions_train))
print(confusion_matrix(y_valid,svm_predictions_train))
acc_svm = accuracy_score(y_valid, svm_predictions_train)
acc_svm
param_grid = {'C': [0.01,0.1,1, 10, 100, 1000,10000], 'gamma': [10,1,0.1,0.01,0.001,0.0001,0.00001], 'kernel': ['rbf']} 
grid_train = GridSearchCV(SVC(),param_grid,refit=True,verbose=10)
grid_train.fit(X_train,y_train) 
grid_train.best_params_   #to inspect the best parameters found by GridSearchCV
grid_train.best_estimator_       #to inspect the best estimator found by GridSearchCV
grid_predictions_train = grid_train.predict(X_valid)
print(classification_report(y_valid,grid_predictions_train))
print(confusion_matrix(y_valid,grid_predictions_train))
acc_grid_svm = accuracy_score(y_valid, grid_predictions_train)
acc_grid_svm
linsvm_train = LinearSVC()
linsvm_train.fit(X_train, y_train)
linsvm_predictions_train = linsvm_train.predict(X_valid)
print(classification_report(y_valid,linsvm_predictions_train))
print(confusion_matrix(y_valid,linsvm_predictions_train))
acc_linsvm = accuracy_score(y_valid, svm_predictions_train)
acc_linsvm
xg_train = XGBClassifier()
xg_train.fit(X_train,y_train)
xg_predictions_train = xg_train.predict(X_valid)
print(classification_report(y_valid,knn_predictions_train))
print(confusion_matrix(y_valid,knn_predictions_train))
acc_xg = accuracy_score(y_valid, xg_predictions_train)
acc_xg
model_performance = pd.DataFrame({
    "Model": ["SVC", "Linear SVC", "Random Forest", 
              "Logistic Regression", "K Nearest Neighbors",  
              "Decision Tree",'XGB','SVC Grid'],
    "Accuracy": [acc_svm, acc_linsvm, acc_rfc, 
              acc_lomo, acc_knn, acc_dtree,acc_xg,acc_grid_svm]
})

model_performance.sort_values(by="Accuracy", ascending=False)
xg_train.fit(X_train,y_train)
submission_of_predictions = xg_train.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": titatest["PassengerId"],
        "Survived": submission_of_predictions
    })

submission.to_csv("titanic.csv", index=False)
print(submission.shape)