import pandas as pd
import seaborn as sb
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        dataset = os.path.join(dirname, filename)
        print(os.path.join(dirname, filename))

data=pd.read_json(dataset)
data1=pd.read_json(dataset)
data.info()
data.tail()
import matplotlib.pyplot as plt
fig,axes = plt.subplots(4,2,figsize=(13,20))
for id,i in enumerate(data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Credit_History','Property_Area', 'Income']]):
    row,col = id//2,id%2
    sb.countplot(x=i,data=data,hue='Application_Status',ax=axes[row,col])
plt.subplots_adjust(hspace=1)
l=data.groupby(data.Dependents)
l.count()
print(round(294/511*100),"% of total applicats are with 0 -dependents")
print(round(85/511*100),"% of total applicats are with 1 -dependents")    
print(round(88/511*100),"% of total applicats are with 2 -dependents")
print(round(44/511*100)," % of total applicats are with 3+ -dependents")
data.groupby(['Self_Employed','Application_Status']).count()
print(46/511*100)
data.groupby(['Married','Gender','Application_Status']).count()
print(87/511*100)
data.groupby(['Property_Area','Application_Status']).count()
fig,axes = plt.subplots(5,2,figsize=(13,20))
for id,i in enumerate(data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Credit_History','Property_Area', 'Income','Application_Status']]):
    row,col = id//2,id%2
    sb.countplot(x=i,data=data,hue='Dependents',ax=axes[row,col])
plt.subplots_adjust(hspace=1)
sb.countplot(hue=data.Dependents,x=data.Income)
data.groupby(['Income','Dependents']).count()
# Average no. of dependents in high income candidates is 
data.groupby(['Income']).count() 
data.groupby(['Application_Status','Credit_History']).count() 
data.head()
data.drop(['Application_ID'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
column=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Income','Application_Status']
all= LabelEncoder()
for i in column:
    data[i] = all.fit_transform(data[i])
data.head()
data.columns
x=data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Credit_History','Property_Area', 'Income']]
y=data.Application_Status
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2, random_state=7)
model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions)*100)
from sklearn.metrics import confusion_matrix #confusuon matrix
pd.crosstab(y_test, predictions)
#positive=1 : application accepted
sb.clustermap(data.corr(),cmap='Blues',annot=True)
#credit history is correlated with application status
X_train=data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Credit_History','Property_Area', 'Income']]
Y_train=data.Application_Status

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3,
                                                    random_state=10)
#Using Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, Y_train)
#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier2.fit(X_train, Y_train)
#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'linear', random_state = 0)
classifier3.fit(X_train, Y_train)
# Using SVC method of svm class to use Kernel SVM Algorithm
from sklearn.svm import SVC
classifier4 = SVC(kernel = 'rbf', random_state = 1)
classifier4.fit(X_train, Y_train)
#Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, Y_train)
#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
classifier6 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier6.fit(X_train, Y_train)
Y_pred1 = classifier1.predict(X_test)
Y_pred2 = classifier2.predict(X_test)
Y_pred3 = classifier3.predict(X_test)
Y_pred4 = classifier4.predict(X_test)
Y_pred5= classifier5.predict(X_test)
Y_pred6 = classifier6.predict(X_test)

print(accuracy_score(Y_test, Y_pred1))
print(accuracy_score(Y_test, Y_pred2))
print(accuracy_score(Y_test, Y_pred3))
print(accuracy_score(Y_test, Y_pred4))
print(accuracy_score(Y_test, Y_pred5))
print(accuracy_score(Y_test, Y_pred6))