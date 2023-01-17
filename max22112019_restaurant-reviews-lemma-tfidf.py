import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#title input file is in tsv format

data=pd.read_csv('../input/reviews/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) 

#delimeter or sep='\t' 

#quoting =3 no quote or ignore quotes while processing
data
import seaborn as sns

sns.countplot(x='Liked', data=data)

plt.show()
good_reviews_count = len(data.loc[data['Liked'] == 1])

bad_reviews_count=len(data.loc[data['Liked']==0])

(good_reviews_count, bad_reviews_count)
#Data Cleaning

# Cleaning the Text

import nltk

import re

nltk.download('stopwords')

nltk.download('wordnet')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer



stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(len(data)):

    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])

    review = review.lower()

    review = review.split()

    all_stopwords = stopwords.words('english')

    all_stopwords.remove('not') 

    #remove negative word 'not' as it is closest word to help determine whether the review is good or not 

    review = [lemmatizer.lemmatize(word) for word in review if not word in set(all_stopwords)]

    review = ' '.join(review)

    corpus.append(review)

print(corpus)
# Creating the TF-IDF model

from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer()

X = cv.fit_transform(corpus).toarray()

y = data.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import VotingClassifier



from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=12)

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
#Logistic Regression

lr = LogisticRegression(C=0.7286427728546842, max_iter=2000, solver='lbfgs', random_state=0)

cv = cross_val_score(lr,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

lr.fit(X_train,y_train)

y_pred_lr=lr.predict(X_test)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(y_pred_lr,y_test)*100)

cm=confusion_matrix(y_test, y_pred_lr)

print(cm)

classification_report(y_test, y_pred_lr)
#GaussianNB

gnb = GaussianNB(var_smoothing=7)

cv = cross_val_score(gnb,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

gnb.fit(X_train,y_train)

y_pred_gnb=gnb.predict(X_test)

print('The accuracy of the Naive Bayes is', metrics.accuracy_score(y_pred_gnb,y_test)*100)

cm=confusion_matrix(y_test, y_pred_gnb)

print(cm)

classification_report(y_test, y_pred_gnb)
#MultinomialNB

mnb = MultinomialNB(alpha=3)

cv = cross_val_score(mnb,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

mnb.fit(X_train,y_train)

y_pred_mnb=mnb.predict(X_test)

print('The accuracy of the Naive Bayes is', metrics.accuracy_score(y_pred_mnb,y_test)*100)

cm=confusion_matrix(y_test, y_pred_mnb)

print(cm)

classification_report(y_test, y_pred_mnb)
#Bernoulli NB

bnb = BernoulliNB(alpha =6)

cv = cross_val_score(bnb,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

mnb.fit(X_train,y_train)

y_pred_bnb=mnb.predict(X_test)

print('The accuracy of the Naive Bayes is', metrics.accuracy_score(y_pred_bnb,y_test)*100)

cm=confusion_matrix(y_test, y_pred_bnb)

print(cm)

classification_report(y_test, y_pred_bnb)
#Random Forest Classifier

rf = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=45,

                       max_features='log2', min_samples_leaf=1,

                       n_estimators=1000, random_state=0)

rf.fit(X_train, y_train)

cv = cross_val_score(rf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_rf = rf.predict(X_test)

print('The accuracy of the RandomForestClassifier is',metrics.accuracy_score(y_pred_rf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_rf)

print(cm)

classification_report(y_test, y_pred_rf)
#Linear SVC

svcl = SVC(kernel = 'linear', random_state = 0, probability=True)

svcl.fit(X_train, y_train)

cv = cross_val_score(svcl,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_svcl = svcl.predict(X_test)

print('The accuracy of the Linear SVC is',metrics.accuracy_score(y_pred_svcl,y_test)*100)

cm=confusion_matrix(y_test, y_pred_svcl)

print(cm)

classification_report(y_test, y_pred_svcl)
#rbf SVC

from sklearn.svm import SVC

svck = SVC(kernel = 'rbf', random_state = 0, probability=True, C=0.62)

svck.fit(X_train, y_train)

cv = cross_val_score(svck,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_svck = svck.predict(X_test)

print('The accuracy of the Kernel SVC is',metrics.accuracy_score(y_pred_svck,y_test)*100)

cm=confusion_matrix(y_test, y_pred_svck)

print(cm)

classification_report(y_test, y_pred_svck)
#Decision Tree Classifier

dt = DecisionTreeClassifier(random_state=0, max_depth=30, min_samples_split=2, min_samples_leaf=1)

dt.fit(X_train, y_train)

cv = cross_val_score(dt,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_dt = dt.predict(X_test)

print('The accuracy of the Decision Tree Classifier is',metrics.accuracy_score(y_pred_dt,y_test)*100)

cm=confusion_matrix(y_test, y_pred_dt)

print(cm)

classification_report(y_test, y_pred_dt)
#KNN

knn = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2, leaf_size = 10)

knn.fit(X_train, y_train)

cv = cross_val_score(knn,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_knn = knn.predict(X_test)

print('The accuracy of the K-Neighbors Classifier is',metrics.accuracy_score(y_pred_knn,y_test)*100)

cm=confusion_matrix(y_test, y_pred_knn)

print(cm)

classification_report(y_test, y_pred_knn)
#VCLF 1

voting_clf = VotingClassifier(estimators = [('lr', lr),('gnb',gnb),('bnb',bnb),('mnb',mnb),

                                            ('knn',knn),('dt',dt),

                                            ('rf',rf),('svck',svck),('svcl',svcl)], voting = 'soft') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)
#VCLF 2

voting_clf = VotingClassifier(estimators = [('lr', lr),('bnb',bnb),('mnb',mnb),('gnb', gnb),

                                            ('rf',rf),('svck',svck),('svcl',svcl)], voting = 'soft') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)
#VCLF 3

voting_clf = VotingClassifier(estimators = [('bnb',bnb),('mnb',mnb),('gnb', gnb),

                                            ('svck',svck),('svcl',svcl)], voting = 'soft') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)
#VCLF 4

voting_clf = VotingClassifier(estimators = [('lr', lr),('gnb',gnb),('bnb',bnb),('mnb',mnb),

                                            ('knn',knn),('dt',dt),

                                            ('rf',rf),('svck',svck),('svcl',svcl)], voting = 'hard') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)
# VCLF 5

voting_clf = VotingClassifier(estimators = [('lr', lr),('bnb',bnb),('mnb',mnb),('gnb', gnb),

                                            ('rf',rf),('svck',svck),('svcl',svcl)], voting = 'hard') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)
# VCLF 6

voting_clf = VotingClassifier(estimators = [('bnb',bnb),('mnb',mnb),('gnb', gnb),

                                            ('svck',svck),('svcl',svcl)], voting = 'hard') 

voting_clf.fit(X_train, y_train)

cv = cross_val_score(voting_clf,X_train,y_train,cv=kfold)

print(cv)

print(cv.mean()*100)

y_pred_vclf = voting_clf.predict(X_test)

print('The accuracy of the Voting Classifier is',metrics.accuracy_score(y_pred_vclf,y_test)*100)

cm=confusion_matrix(y_test, y_pred_vclf)

print(cm)

classification_report(y_test, y_pred_vclf)