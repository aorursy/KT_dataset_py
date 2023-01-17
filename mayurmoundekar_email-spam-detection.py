import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
spam = pd.read_csv('../input/email-spam-detection-using-machine-learning/email spam.csv')

spam.head()
spam.shape
spam.describe()
sns.countplot(data = spam, x= spam["Label"]).set_title("Amount of spam and no-spam messages")

plt.show()
count_Class=pd.value_counts(spam.Label, sort= True)



# Data to plot

labels = 'Ham', 'Spam'

sizes = [count_Class[0], count_Class[1]]

colors = ['gold', 'yellowgreen'] # 'lightcoral', 'lightskyblue'

explode = (0.1, 0.1)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()
X = spam["EmailText"]

X.head()
y = spam["Label"]

y.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()  

trainCV = cv.fit_transform(X_train)

testCV = cv.transform(X_test)
from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()

naive_bayes.fit(trainCV,y_train)

pred_NB = naive_bayes.predict(testCV)
from sklearn.metrics import accuracy_score

Accuracy_Score_NB = accuracy_score(y_test, pred_NB)

Accuracy_Score_NB
from sklearn.neighbors import KNeighborsClassifier

classifier_knn = KNeighborsClassifier()

classifier_knn.fit(trainCV, y_train)

pred_knn = classifier_knn.predict(testCV)
Accuracy_Score_knn = accuracy_score(y_test, pred_knn)

Accuracy_Score_knn
from sklearn.svm import SVC

classifier_svm_linear = SVC(kernel = 'linear')

classifier_svm_linear.fit(trainCV, y_train)

pred_svm_linear = classifier_svm_linear.predict(testCV)
Accuracy_Score_SVM_Linear = accuracy_score(y_test, pred_svm_linear)

Accuracy_Score_SVM_Linear
classifier_svm_rbf = SVC(kernel = 'rbf')

classifier_svm_rbf.fit(trainCV, y_train)

pred_svm_rbf = classifier_svm_rbf.predict(testCV)
Accuracy_Score_SVM_Gaussion = accuracy_score(y_test, pred_svm_rbf)

Accuracy_Score_SVM_Gaussion
from sklearn.tree import DecisionTreeClassifier

classifier_dt = DecisionTreeClassifier()

classifier_dt.fit(trainCV, y_train)

pred_dt = classifier_dt.predict(testCV)
Accuracy_Score_dt = accuracy_score(y_test, pred_dt)

Accuracy_Score_dt
from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier()

classifier_rf.fit(trainCV, y_train)

pred_rf = classifier_rf.predict(testCV)
Accuracy_Score_rf = accuracy_score(y_test, pred_rf)

Accuracy_Score_rf
import xgboost as xgb

classifier_xg = xgb.XGBClassifier()

classifier_xg.fit(trainCV, y_train)

pred_xg = classifier_xg.predict(testCV)
Accuracy_Score_xg = accuracy_score(y_test, pred_xg)

Accuracy_Score_xg
print("K-Nearest Neighbors =",Accuracy_Score_knn)

print("Naive Bayes =",Accuracy_Score_NB)

print("Support Vector Machine Linear =",Accuracy_Score_SVM_Linear)

print("Support Vector Machine Gaussion =",Accuracy_Score_SVM_Gaussion)

print("Decision Tree =",Accuracy_Score_dt)

print("Random Forest =",Accuracy_Score_rf)

print("XgBoost =",Accuracy_Score_xg)