# In this notebook, we will use the polarity and length (measured as the number of words)

# of the reviews as predictor variables for the quality of a wine. We will try out several

# classifiers and calculate the metrics that give us a taste of their performance

from __future__ import unicode_literals

from textblob import TextBlob

import pandas as pd

import nltk

import numpy as np

wine = pd.read_csv("../input/winemag-data_first150k.csv",sep=",")
wine = wine.drop_duplicates()

wine= wine.dropna()
def polarity_function(review):#measures polarity of wine description

    opinion_wine=TextBlob(review)

    return opinion_wine.sentiment.polarity



def subjectivity_function(review): #measures subjectivity of wine description

    opinion_wine=TextBlob(review)

    return opinion_wine.sentiment.subjectivity



def words_function(review): # measures the length of the wine description by number of words

    t=TextBlob(review)

    return len(t.words)
wine['polarity']= wine.description.apply(polarity_function)
wine['subjectivity']= wine.description.apply(subjectivity_function)
wine['num_words'] = wine.description.apply(words_function)
def rating_type(score):

    if score > 88:

        return 1

    if score <= 88:

        return 0

#creating a binary variable named "quality" based on wine rating

wine['quality'] = wine.points.apply(rating_type)
wine_f = wine[['num_words','polarity','quality']]
wine_f.head()
wine_f.corr()
# let´s plot the data in the plane (number of words,polarity). 

#The color represents the quality label

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

plt.scatter(wine["num_words"],wine["polarity"],c=wine["quality"],s=6)

plt.xlabel('number of words in description',fontsize=14)

plt.ylabel('description polarity',fontsize=14)

plt.show()
# the number of words in the description is highly correlated with the wine`s quality

X = wine[['polarity','subjectivity','num_words']]

X=X.values
y = wine['quality']
# logistic regression
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
clf_lr.fit(X_train,y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
np.mean(cross_val_score(clf_lr,X,y,cv=50))*100
np.std(cross_val_score(clf_lr,X,y,cv=50))*100
import matplotlib.pyplot as plt

%matplotlib inline
plt.hist(cross_val_score(clf_lr,X_test,y_test,cv=50))
precision_score(clf_lr.predict(X_test),y_test)
recall_score(clf_lr.predict(X_test),y_test)
confusion_matrix(clf_lr.predict(X_test),y_test)
from sklearn.metrics import roc_curve, auc

from sklearn.cross_validation import train_test_split

 

# shuffle and split training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

clf_lr.fit(X_train, y_train)

 

# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(y_test, clf_lr.predict_proba(X_test)[:,1])

 

# Calculate the AUC

roc_auc = auc(fpr, tpr)

print(roc_auc)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve for Logistic Regression')

plt.legend(loc="lower right")

plt.show()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC
clf_knn = KNeighborsClassifier(n_neighbors=19)
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 42)
clf_knn.fit(X_train,y_train)
np.mean(cross_val_score(clf_knn,X_test,y_test,cv=50))*100
np.std(cross_val_score(clf_knn,X_test,y_test,cv=50))*100
#finding the most appropriate number of neighbors

acc_list=[]

for n in range(1,50):

    clf_knn = KNeighborsClassifier(n_neighbors=n)

    clf_knn.fit(X_train,y_train)

    mean_acc = np.mean(cross_val_score(clf_knn,X_test,y_test,cv=50))*100

    acc_list.append(mean_acc)
plt.figure(figsize=(10,10))

plt.plot(range(1,50),acc_list)

plt.show()
precision_score(clf_knn.predict(X_test),y_test)
recall_score(clf_knn.predict(X_test),y_test)
confusion_matrix(clf_lr.predict(X_test),y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

clf_knn.fit(X_train, y_train)

 

# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(y_test, clf_knn.predict_proba(X_test)[:,1])

 

# Calculate the AUC

roc_auc = auc(fpr, tpr)

print(roc_auc)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve for K-nearest neighbors')

plt.legend(loc="lower right")

plt.show()
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=2,oob_score=True,random_state=42)
clf_rf.fit(X_train,y_train)
# shuffle and split training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

clf_rf.fit(X_train, y_train)

 

# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(y_test, clf_knn.predict_proba(X_test)[:,1])

 

# Calculate the AUC

roc_auc = auc(fpr, tpr)

print(roc_auc)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve for Random Forest')

plt.legend(loc="lower right")

plt.show()
precision_score(clf_rf.predict(X_test),y_test)
recall_score(clf_rf.predict(X_test),y_test)
confusion_matrix(clf_rf.predict(X_test),y_test)
confusion_matrix(clf_lr.predict(X_test),y_test)
confusion_matrix(clf_knn.predict(X_test),y_test)
clf_svm = LinearSVC(C=.1)

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
clf_svm.fit(X_train,y_train)
clf_svm.fit(X_train, y_train)

 

# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(y_test, clf_knn.predict_proba(X_test)[:,1])

 

# Calculate the AUC

roc_auc = auc(fpr, tpr)

print(roc_auc)

 

# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve for Support Vector Machine')

plt.legend(loc="lower right")

plt.show()
confusion_matrix(clf_svm.predict(X_test),y_test)
accuracy_score(clf_svm.predict(X_test),y_test)
precision_score(clf_svm.predict(X_test),y_test)
recall_score(clf_svm.predict(X_test),y_test)
plt.hist(cross_val_score(clf_svm,X_test,y_test,cv=50))