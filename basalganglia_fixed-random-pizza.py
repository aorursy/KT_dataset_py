import numpy as np
import pandas as pd

# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

pd_train = pd.read_json('../input/train.json', orient='columns')
pd_test = pd.read_json('../input/test.json', orient='columns')


np_test = np.array(pd_test)
np_train = np.array(pd_train)

print(np_train.shape)


X = np_train[:,6]
Y = np_train[:,22]
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]

print('data shape: ', X.shape)
print('label shape:', Y.shape)

#l=len(X)
l = int(4040)
train_data, train_labels = X[:int(l/2)], Y[:int(l/2)]
dev_data, dev_labels = X[int(l/2):int((3*l)/4)], Y[int(l/2):int((3*l)/4)]
test_data, test_labels = X[int((3*l)/4):], Y[int((3*l)/4):]

# Any results you write to the current directory are saved as output.
l = len(X)
l
vect=CountVectorizer()
data=vect.fit_transform(train_data).toarray()
devdata=vect.transform(dev_data).toarray()
data.shape
#Use np.where to binarize train and dev set where values above and below 0.5.
b=train_labels
trainlabels=np.where(b==True, 1, 0)

bl=dev_labels
devlabels=np.where(bl==True, 1, 0)

b2=test_labels
testlabels=np.where(b2==True, 1, 0)

categories = ['Got Pizza', 'Didn\'t get pizza']

print('Baseline Scores...')
#Run MultinomialNB Classifier
# mnb_clf = Pipeline([('vect', CountVectorizer()), ('mnclf',MultinomialNB(alpha=0.01))])
# mnb_clf = mnb_clf.fit(train_data, trainlabels)
# pred = mnb_clf.predict(dev_data)
# score1=metrics.accuracy_score(devlabels,pred)
# print 'Naive Bayes Score:',score1
best_nb = []
alphas = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
for k in range(len(alphas)):
    mnb_clf = Pipeline([('vect', CountVectorizer()), ('mnclf', MultinomialNB(alpha=alphas[k]))])
    mnb_clf = mnb_clf.fit(train_data, trainlabels)
    pred = mnb_clf.predict(dev_data)
    metrics.accuracy_score(devlabels,pred)
    best_nb.append(metrics.accuracy_score(devlabels,pred))
bestAlphaAccuracy = max(best_nb)
bestAlphaValue = alphas[best_nb.index(bestAlphaAccuracy)]
print('Naive Bayes Baseline:')
print('Best Alpha =', bestAlphaValue, ' accuracy:', bestAlphaAccuracy)
print('')



#Run Logistic Regression classifier
log_clf = Pipeline([('vect', CountVectorizer()),('lgclf', LogisticRegression(C=0.5))])
log_clf = log_clf.fit(train_data, trainlabels) 
pred = log_clf.predict(dev_data)        
score2= metrics.accuracy_score(devlabels,pred)
#print 'Logistic Regression Score:',score2
best_logit = []
#C = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
C= [0.8]
for k in range(len(C)):
    log_clf = Pipeline([('vect', CountVectorizer()),
                     ('lgclf', LogisticRegression(C=C[k]))]);
    log_clf = log_clf.fit(train_data, trainlabels)
    pred = log_clf.predict(dev_data)
    metrics.accuracy_score(devlabels,pred)
    best_logit.append(metrics.accuracy_score(devlabels,pred))
    weights = log_clf.named_steps['lgclf'].coef_
bestCAccuracy = max(best_logit)
bestCValue = C[best_logit.index(bestCAccuracy)]
print('Logistic Regression Baseline:')
print('Best C =', bestCValue, ' accuracy:', bestCAccuracy)
print('')
