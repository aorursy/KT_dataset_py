import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics

df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness","insulin","bmi","Diabetes_Pedigree_Function","age","outcome"]

df.head(5)
from sklearn.preprocessing import scale
df['insulin'] = scale(df['insulin'])

#logistic regression algo

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
y = df['outcome']
x = df.drop('outcome',axis =1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_train)
LR=model.score(x_test,y_test)
print("Accuracy of Logistic Regression: ", LR)



#KNN algorithm

#find the closed KNN value

import matplotlib.pyplot as plt
plt.style.use('ggplot')

#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,25)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(x_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(x_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(x_test, y_test) 
    
    plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# knn 20 and 23 looks better accuracy

from sklearn.neighbors import KNeighborsClassifier
knn1 = KNeighborsClassifier(20)
knn1.fit(x_train,y_train)
knn1=knn1.score(x_test,y_test)
print("Accuracy of KNN with 20: ", knn1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(23)
knn.fit(x_train,y_train)
knn=knn.score(x_test,y_test)
print("Accuracy of KNN with 23: ", knn)


#Decision Tree Classfifier

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier = classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
DTR=metrics.accuracy_score(y_test,y_pred)
print("Accuracy of Decision Tree Classifier : ", DTR)




#Naive Bayes

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
#cm = confusion_matrix(y_test, y_pred)

NB=accuracy_score(y_test, y_pred)
print("Accuracy of Naive Bayes: ", NB)

#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, random_state = 0) 
forest.fit(x_train, y_train)
RF=forest.score(x_test, y_test)

#cm_rf = confusion_matrix(y_test, y_pred)
print("Accuracy of Random Forest Classifier: ", RF)

#Support Vector Machine

from sklearn.svm import SVC
svc = SVC(kernel = "linear")
svc.fit(x_train, y_train)
svm=forest.score(x_test,y_test)
print("Accuracy of Support Vector Machine: ", svm)

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
print(cm)





