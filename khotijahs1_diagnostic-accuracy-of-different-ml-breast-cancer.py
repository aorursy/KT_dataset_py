import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')

print('Dataset :',data.shape)

x = data.iloc[:, [0, 1, 2, 3]].values

data.info()

data[0:10]
data.diagnosis.value_counts()[0:30].plot(kind='bar')

plt.show()
data = data[['mean_radius','mean_texture','mean_perimeter','mean_area', 'mean_smoothness','diagnosis']] #Subsetting the data

cor = data.corr() #Calculate the correlation of the above variables

sns.heatmap(cor, square = True) #Plot the correlation as heat map
sns.set_style("ticks")

sns.pairplot(data,hue="diagnosis",size=3);

plt.show()
from sklearn.model_selection import train_test_split

Y = data['diagnosis']

X = data.drop(columns=['diagnosis'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=9)
print('X train shape: ', X_train.shape)

print('Y train shape: ', Y_train.shape)

print('X test shape: ', X_test.shape)

print('Y test shape: ', Y_test.shape)
from sklearn.linear_model import LogisticRegression



# We defining the model

logreg = LogisticRegression(C=10)



# We train the model

logreg.fit(X_train, Y_train)



# We predict target values

Y_predict1 = logreg.predict(X_test)
# The confusion matrix

from sklearn.metrics import confusion_matrix

import seaborn as sns



logreg_cm = confusion_matrix(Y_test, Y_predict1)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(logreg_cm, annot=True, linewidth=0.7, linecolor='red', fmt='g', ax=ax, cmap="BuPu")

plt.title('Logistic Regression Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_logreg = logreg.score(X_test, Y_test)

print(score_logreg)
#precision and recall

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict1)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.ensemble import BaggingClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC



# We define the SVM model

svmcla = OneVsRestClassifier(BaggingClassifier(SVC(C=10,kernel='rbf',random_state=9, probability=True), 

                                               n_jobs=-1))



# We train model

svmcla.fit(X_train, Y_train)



# We predict target values

Y_predict2 = svmcla.predict(X_test)
# The confusion matrix

svmcla_cm = confusion_matrix(Y_test, Y_predict2)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(svmcla_cm, annot=True, linewidth=0.7, linecolor='red', fmt='g', ax=ax, cmap="BuPu")

plt.title('SVM Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_svmcla = svmcla.score(X_test, Y_test)

print(score_svmcla)
#precision and recall

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict2)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.naive_bayes import GaussianNB



# We define the model

nbcla = GaussianNB()



# We train model

nbcla.fit(X_train, Y_train)



# We predict target values

Y_predict3 = nbcla.predict(X_test)
# The confusion matrix

nbcla_cm = confusion_matrix(Y_test, Y_predict3)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(nbcla_cm, annot=True, linewidth=0.7, linecolor='red', fmt='g', ax=ax, cmap="BuPu")

plt.title('Naive Bayes Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_nbcla = nbcla.score(X_test, Y_test)

print(score_nbcla)
#precision and recall

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict3)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.tree import DecisionTreeClassifier



# We define the model

dtcla = DecisionTreeClassifier(random_state=9)



# We train model

dtcla.fit(X_train, Y_train)



# We predict target values

Y_predict4 = dtcla.predict(X_test)
# The confusion matrix

dtcla_cm = confusion_matrix(Y_test, Y_predict4)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(dtcla_cm, annot=True, linewidth=0.7, linecolor='red', fmt='g', ax=ax, cmap="BuPu")

plt.title('Decision Tree Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_dtcla = dtcla.score(X_test, Y_test)

print(score_dtcla)
#precision and recall

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict4)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.ensemble import RandomForestClassifier



# We define the model

rfcla = RandomForestClassifier(n_estimators=100,random_state=9,n_jobs=-1)



# We train model

rfcla.fit(X_train, Y_train)



# We predict target values

Y_predict5 = rfcla.predict(X_test)
# The confusion matrix

rfcla_cm = confusion_matrix(Y_test, Y_predict5)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(rfcla_cm, annot=True, linewidth=0.7, linecolor='red', fmt='g', ax=ax, cmap="BuPu")

plt.title('Random Forest Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_rfcla = rfcla.score(X_test, Y_test)

print(score_rfcla)
#precision and recall

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict5)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.neighbors import KNeighborsClassifier



# We define the model

knncla = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)



# We train model

knncla.fit(X_train, Y_train)



# We predict target values

Y_predict6 = knncla.predict(X_test)
# The confusion matrix

knncla_cm = confusion_matrix(Y_test, Y_predict6)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(knncla_cm, annot=True, linewidth=0.7, linecolor='red', fmt='g', ax=ax, cmap="BuPu")

plt.title('KNN Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_knncla= knncla.score(X_test, Y_test)

print(score_knncla)
#precision and recall

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict6)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
Testscores = pd.Series([score_logreg, score_svmcla, score_nbcla, score_dtcla, score_rfcla, score_knncla], 

                        index=['Logistic Regression Score', 'Support Vector Machine Score', 'Naive Bayes Score', 'Decision Tree Score', 'Random Forest Score', 'K-Nearest Neighbour Score']) 

print(Testscores)
fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3, 3, 1) 

ax1.set_title('Logistic Regression Classification') 

ax2 = fig.add_subplot(3, 3, 2) 

ax2.set_title('SVM Classification')

ax3 = fig.add_subplot(3, 3, 3)

ax3.set_title('Naive Bayes Classification')

ax4 = fig.add_subplot(3, 3, 4)

ax4.set_title('Decision Tree Classification')

ax5 = fig.add_subplot(3, 3, 5)

ax5.set_title('Random Forest Classification')

ax6 = fig.add_subplot(3, 3, 6)

ax6.set_title('KNN Classification')

sns.heatmap(data=logreg_cm, annot=True, linewidth=0.7, linecolor='red',cmap="BuPu" ,fmt='g', ax=ax1)

sns.heatmap(data=svmcla_cm, annot=True, linewidth=0.7, linecolor='red',cmap="BuPu" ,fmt='g', ax=ax2)  

sns.heatmap(data=nbcla_cm, annot=True, linewidth=0.7, linecolor='red',cmap="BuPu" ,fmt='g', ax=ax3)

sns.heatmap(data=dtcla_cm, annot=True, linewidth=0.7, linecolor='red',cmap="BuPu" ,fmt='g', ax=ax4)

sns.heatmap(data=rfcla_cm, annot=True, linewidth=0.7, linecolor='red',cmap="BuPu" ,fmt='g', ax=ax5)

sns.heatmap(data=knncla_cm, annot=True, linewidth=0.7, linecolor='red',cmap="BuPu" ,fmt='g', ax=ax6)

plt.show()
from sklearn.metrics import roc_curve



# Logistic Regression Classification

Y_predict1_proba = logreg.predict_proba(X_test)

Y_predict1_proba = Y_predict1_proba[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, Y_predict1_proba)

plt.subplot(331)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC Curve Logistic Regression')

plt.grid(True)



# SVM Classification

Y_predict2_proba = svmcla.predict_proba(X_test)

Y_predict2_proba = Y_predict2_proba[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, Y_predict2_proba)

plt.subplot(332)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC Curve SVM')

plt.grid(True)



# Naive Bayes Classification

Y_predict3_proba = nbcla.predict_proba(X_test)

Y_predict3_proba = Y_predict3_proba[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, Y_predict3_proba)

plt.subplot(333)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC Curve Naive Bayes')

plt.grid(True)



# Decision Tree Classification

Y_predict4_proba = dtcla.predict_proba(X_test)

Y_predict4_proba = Y_predict4_proba[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, Y_predict4_proba)

plt.subplot(334)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC Curve Decision Tree')

plt.grid(True)



# Random Forest Classification

Y_predict5_proba = rfcla.predict_proba(X_test)

Y_predict5_proba = Y_predict5_proba[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, Y_predict5_proba)

plt.subplot(335)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC Curve Random Forest')

plt.grid(True)



# KNN Classification

Y_predict6_proba = knncla.predict_proba(X_test)

Y_predict6_proba = Y_predict6_proba[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, Y_predict6_proba)

plt.subplot(336)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC Curve KNN')

plt.grid(True)

plt.subplots_adjust(top=2, bottom=0.08, left=0.10, right=1.4, hspace=0.45, wspace=0.45)

plt.show()