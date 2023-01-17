import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('../input/lung-cancer-dataset/lung_cancer_examples.csv')

print('Dataset :',data.shape)

data.info()

data[0:10]
# Distribution of diagnosis

data.Result.value_counts()[0:30].plot(kind='bar')

plt.show()
sns.set_style("whitegrid")

sns.pairplot(data,hue="Result",size=3);

plt.show()
data1 = data.drop(columns=['Name','Surname'],



                 axis=1)

data1 = data1.dropna(how='any')

print(data1.shape)
print(data1.shape)

data1.head()
from sklearn.model_selection import train_test_split

Y = data1['Result']

X = data1.drop(columns=['Result'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=9)
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

sns.heatmap(logreg_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")

plt.title('Logistic Regression Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_logreg = logreg.score(X_test, Y_test)

print(score_logreg)
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

sns.heatmap(svmcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")

plt.title('SVM Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_svmcla = svmcla.score(X_test, Y_test)

print(score_svmcla)
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

sns.heatmap(nbcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")

plt.title('Naive Bayes Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_nbcla = nbcla.score(X_test, Y_test)

print(score_nbcla)
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

sns.heatmap(dtcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")

plt.title('Decision Tree Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_dtcla = dtcla.score(X_test, Y_test)

print(score_dtcla)
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

sns.heatmap(rfcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")

plt.title('Random Forest Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_rfcla = rfcla.score(X_test, Y_test)

print(score_rfcla)
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

sns.heatmap(knncla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")

plt.title('KNN Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_knncla= knncla.score(X_test, Y_test)

print(score_knncla)
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

sns.heatmap(data=logreg_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax1)

sns.heatmap(data=svmcla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax2)  

sns.heatmap(data=nbcla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax3)

sns.heatmap(data=dtcla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax4)

sns.heatmap(data=rfcla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax5)

sns.heatmap(data=knncla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax6)

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
Y1 = data1['Result']

X1 = data1.drop(columns=['Age'])

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel



lsvc = LinearSVC(C=0.06, penalty="l1", dual=False,random_state=10).fit(X1, Y1)

model = SelectFromModel(lsvc, prefit=True)

X_new = model.transform(X1)

cc = list(X1.columns[model.get_support(indices=True)])

print(cc)

print(len(cc))
# Principal component analysis

from sklearn.decomposition import PCA



pca = PCA().fit(X1)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Smokes AreaQ Alkhol Result')

plt.ylabel('% Variance Explained')

plt.title('PCA Analysis')

plt.grid(True)

plt.show()
# Percentage of total variance explained

variance = pd.Series(list(np.cumsum(pca.explained_variance_ratio_)), 

                        index= list(range(1,5))) 

print(variance[10:90])
X1 = data1[cc] 

from sklearn.model_selection import train_test_split

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.1, random_state=10)
# Logistic regression classification

logreg.fit(X1_train, Y1_train)

Y1_predict1 = logreg.predict(X1_test)

logreg_cm = confusion_matrix(Y1_test, Y1_predict1)

score1_logreg = logreg.score(X1_test, Y1_test)



# SVM classification

svmcla.fit(X1_train, Y1_train)

Y1_predict2 = svmcla.predict(X1_test)

svmcla_cm = confusion_matrix(Y1_test, Y1_predict2)

score1_svmcla = svmcla.score(X1_test, Y1_test)



# Random forest classification

rfcla.fit(X1_train, Y1_train)

Y1_predict5 = rfcla.predict(X1_test)

rfcla_cm = confusion_matrix(Y1_test, Y1_predict5)

score1_rfcla = rfcla.score(X1_test, Y1_test)
Testscores1 = pd.Series([score1_logreg, score1_svmcla, score1_rfcla], index=['Logistic Regression Score', 

                        'Support Vector Machine Score', 'Random Forest Score']) 

print(Testscores1)
fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3, 3, 1) 

ax1.set_title('Logistic Regression Classification') 

ax2 = fig.add_subplot(3, 3, 2) 

ax2.set_title('SVM Classification')

ax5 = fig.add_subplot(3, 3,3)

ax5.set_title('Random Forest Classification')



sns.heatmap(data=logreg_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax1)

sns.heatmap(data=svmcla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax2)  

sns.heatmap(data=rfcla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax5)

plt.show()
Y1 = data1['Result']

X1 = data1.drop(columns=['AreaQ'])

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel



lsvc = LinearSVC(C=0.06, penalty="l1", dual=False,random_state=10).fit(X1, Y1)

model = SelectFromModel(lsvc, prefit=True)

X_new = model.transform(X1)

cc = list(X1.columns[model.get_support(indices=True)])

print(cc)

print(len(cc))
# Principal component analysis

from sklearn.decomposition import PCA



pca = PCA().fit(X1)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Age Smokes Alkhol Result')

plt.ylabel('% Variance Explained')

plt.title('PCA Analysis')

plt.grid(True)

plt.show()
# Percentage of total variance explained

variance = pd.Series(list(np.cumsum(pca.explained_variance_ratio_)), 

                        index= list(range(1,5))) 

print(variance[10:90])
X1 = data1[cc] 

from sklearn.model_selection import train_test_split

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.1, random_state=10)
# Logistic regression classification

logreg.fit(X1_train, Y1_train)

Y1_predict1 = logreg.predict(X1_test)

logreg_cm = confusion_matrix(Y1_test, Y1_predict1)

score1_logreg = logreg.score(X1_test, Y1_test)



# SVM classification

svmcla.fit(X1_train, Y1_train)

Y1_predict2 = svmcla.predict(X1_test)

svmcla_cm = confusion_matrix(Y1_test, Y1_predict2)

score1_svmcla = svmcla.score(X1_test, Y1_test)



# Random forest classification

rfcla.fit(X1_train, Y1_train)

Y1_predict5 = rfcla.predict(X1_test)

rfcla_cm = confusion_matrix(Y1_test, Y1_predict5)

score1_rfcla = rfcla.score(X1_test, Y1_test)
Testscores1 = pd.Series([score1_logreg, score1_svmcla, score1_rfcla], index=['Logistic Regression Score', 

                        'Support Vector Machine Score', 'Random Forest Score']) 

print(Testscores1)
fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3, 3, 1) 

ax1.set_title('Logistic Regression Classification') 

ax2 = fig.add_subplot(3, 3, 2) 

ax2.set_title('SVM Classification')

ax5 = fig.add_subplot(3, 3,3)

ax5.set_title('Random Forest Classification')

sns.heatmap(data=logreg_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax1)

sns.heatmap(data=svmcla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax2)  

sns.heatmap(data=rfcla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax5)

plt.show()
Y1 = data1['Result']

X1 = data1.drop(columns=['Alkhol'])

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel



lsvc = LinearSVC(C=0.06, penalty="l1", dual=False,random_state=10).fit(X1, Y1)

model = SelectFromModel(lsvc, prefit=True)

X_new = model.transform(X1)

cc = list(X1.columns[model.get_support(indices=True)])

print(cc)

print(len(cc))
# Principal component analysis

from sklearn.decomposition import PCA



pca = PCA().fit(X1)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Age Smokes AreaQ Result')

plt.ylabel('% Variance Explained')

plt.title('PCA Analysis')

plt.grid(True)

plt.show()
# Percentage of total variance explained

variance = pd.Series(list(np.cumsum(pca.explained_variance_ratio_)), 

                        index= list(range(1,5))) 

print(variance[10:90])
X1 = data1[cc] 

from sklearn.model_selection import train_test_split

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.1, random_state=10)
# Logistic regression classification

logreg.fit(X1_train, Y1_train)

Y1_predict1 = logreg.predict(X1_test)

logreg_cm = confusion_matrix(Y1_test, Y1_predict1)

score1_logreg = logreg.score(X1_test, Y1_test)



# SVM classification

svmcla.fit(X1_train, Y1_train)

Y1_predict2 = svmcla.predict(X1_test)

svmcla_cm = confusion_matrix(Y1_test, Y1_predict2)

score1_svmcla = svmcla.score(X1_test, Y1_test)



# Random forest classification

rfcla.fit(X1_train, Y1_train)

Y1_predict5 = rfcla.predict(X1_test)

rfcla_cm = confusion_matrix(Y1_test, Y1_predict5)

score1_rfcla = rfcla.score(X1_test, Y1_test)
Testscores1 = pd.Series([score1_logreg, score1_svmcla, score1_rfcla], index=['Logistic Regression Score', 

                        'Support Vector Machine Score', 'Random Forest Score']) 

print(Testscores1)
fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3, 3, 1) 

ax1.set_title('Logistic Regression Classification') 

ax2 = fig.add_subplot(3, 3, 2) 

ax2.set_title('SVM Classification')

ax5 = fig.add_subplot(3, 3,3)

ax5.set_title('Random Forest Classification')

sns.heatmap(data=logreg_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax1)

sns.heatmap(data=svmcla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax2)  

sns.heatmap(data=rfcla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="YlGnBu" ,fmt='g', ax=ax5)

plt.show()