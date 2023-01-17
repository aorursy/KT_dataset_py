import pandas as pd

from sklearn.metrics import confusion_matrix

mydataset = pd.read_csv('../input/news.csv') 

X = mydataset.iloc[:,1]#taking all rows and title column from dataset

y = mydataset.iloc[:,4]#taking all rows and category column from dataset



from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()

vect.fit(X_train)

X_train_dtm = vect.transform(X_train)

X_train_dtm = vect.fit_transform(X_train)

X_test_dtm = vect.transform(X_test)
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)



y_pred_class  = nb.predict(X_test_dtm)



from sklearn import metrics

print('Accuracy Precentage with Naive Bayes',metrics.accuracy_score(y_test,y_pred_class)*100)

print("Confusion Matrix of Naive Bayes ",metrics.confusion_matrix(y_test,y_pred_class))

from sklearn.tree import DecisionTreeClassifier 

DTC = DecisionTreeClassifier()

DTC.fit(X_train_dtm, y_train)

y1_pred_class = DTC.predict(X_test_dtm)

print('Accuracy Precentage with Decision tree classifier',metrics.accuracy_score(y_test,y1_pred_class)*100)

print("Confusion Matrix of Decision Tree Classifier ",metrics.confusion_matrix(y_test,y1_pred_class))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train_dtm, y_train)

y3_pred_class = knn.predict(X_test_dtm)

print('Accuracy Precentage with KNN',metrics.accuracy_score(y_test,y3_pred_class)*100)

print("Confusion Matrix of KNN ",metrics.confusion_matrix(y_test,y3_pred_class))