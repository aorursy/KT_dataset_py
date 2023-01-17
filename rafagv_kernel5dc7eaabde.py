import pandas as pd
import numpy as np
import sklearn
import math
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, make_scorer
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
score = make_scorer(fbeta_score, beta=3)
df = pd.read_csv("../input/spamdataset/train_data.csv",
    sep=r'\s*,\s*',
    engine='python',
    na_values="None")
test = pd.read_csv("../input/spamdataset/test_features.csv",
    sep=r'\s*,\s*',
    engine='python',
    na_values="None")
sid = test.Id.values
test = test.drop('Id', axis = 1)
Y = df.ham
D = df.drop("ham", axis = 1)
D = D.drop("Id", axis = 1)
X = df.drop("ham", axis = 1)
X = X.drop("Id", axis = 1)
X = StandardScaler().fit_transform(X)
M = df.mean()
A = []
z = -1
for j in df.columns:
    DF = df[j]
    z += 1
    A.append([])
    N = (M[z])
    for i in range(3680):
        if(DF[i] > N):
            A[z].append(1)
        else: 
            A[z].append(0)
len(A)
z = 0
for i in df.columns:
    df[i] = A[z]
    z += 1
M = df.mean()
A = []
z = -1
for j in test.columns:
    DF = test[j]
    z += 1
    A.append([])
    N = (M[z])
    for i in range(921):
        if(DF[i] > N):
            A[z].append(1)
        else: 
            A[z].append(0)
z = 0
for i in test.columns:
    test[i] = A[z]
    z += 1
len(A)
D = df.drop("ham", axis = 1)
D = D.drop("Id", axis = 1)
Dtest = test
C = df.corr()
C
D = D.drop('char_freq_[', axis = 1)
D = D.drop('char_freq_;', axis = 1)
D = D.drop('char_freq_(', axis = 1)
D = D.drop('word_freq_table', axis = 1)
D = D.drop('word_freq_original', axis = 1)
D = D.drop('word_freq_direct', axis = 1)
D = D.drop('word_freq_parts', axis = 1)
D = D.drop('word_freq_report', axis = 1)
D = D.drop('word_freq_will', axis = 1)
D = D.drop('word_freq_conference', axis = 1)
D = D.drop('word_freq_re', axis = 1)
D = D.drop('word_freq_project', axis = 1)
D = D.drop('word_freq_cs', axis = 1)
D = D.drop('word_freq_pm', axis = 1)
D = D.drop('char_freq_#', axis = 1)
D = D.drop('word_freq_meeting', axis = 1)
D = D.drop('word_freq_technology', axis = 1)
D = D.drop('word_freq_415', axis = 1)
D = D.drop('word_freq_data', axis = 1)
D = D.drop('word_freq_857', axis = 1)
#menor que 29
D = D.drop('word_freq_make', axis = 1)
D = D.drop('word_freq_address', axis = 1)
D = D.drop('word_freq_mail', axis = 1)
D = D.drop('word_freq_people', axis = 1)
D = D.drop('word_freq_addresses', axis = 1)
D = D.drop('word_freq_email', axis = 1)
D = D.drop('word_freq_650', axis = 1)
D = D.drop('word_freq_lab', axis = 1)
D = D.drop('word_freq_labs', axis = 1)
D = D.drop('word_freq_telnet', axis = 1)
D = D.drop('word_freq_85', axis = 1)
D = D.drop('word_freq_1999', axis = 1)
D = D.drop('word_freq_edu', axis = 1)

Dtest = Dtest.drop('char_freq_[', axis = 1)
Dtest = Dtest.drop('char_freq_;', axis = 1)
Dtest = Dtest.drop('char_freq_(', axis = 1)
Dtest = Dtest.drop('word_freq_table', axis = 1)
Dtest = Dtest.drop('word_freq_original', axis = 1)
Dtest = Dtest.drop('word_freq_direct', axis = 1)
Dtest = Dtest.drop('word_freq_parts', axis = 1)
Dtest = Dtest.drop('word_freq_report', axis = 1)
Dtest = Dtest.drop('word_freq_will', axis = 1)
Dtest = Dtest.drop('word_freq_conference', axis = 1)
Dtest = Dtest.drop('word_freq_re', axis = 1)
Dtest = Dtest.drop('word_freq_project', axis = 1)
Dtest = Dtest.drop('word_freq_cs', axis = 1)
Dtest = Dtest.drop('word_freq_pm', axis = 1)
Dtest = Dtest.drop('char_freq_#', axis = 1)
Dtest = Dtest.drop('word_freq_meeting', axis = 1)
Dtest = Dtest.drop('word_freq_technology', axis = 1)
Dtest = Dtest.drop('word_freq_415', axis = 1)
Dtest = Dtest.drop('word_freq_data', axis = 1)
Dtest = Dtest.drop('word_freq_857', axis = 1)
#menor que 29
Dtest = Dtest.drop('word_freq_make', axis = 1)
Dtest = Dtest.drop('word_freq_address', axis = 1)
Dtest = Dtest.drop('word_freq_mail', axis = 1)
Dtest = Dtest.drop('word_freq_people', axis = 1)
Dtest = Dtest.drop('word_freq_addresses', axis = 1)
Dtest = Dtest.drop('word_freq_email', axis = 1)
Dtest = Dtest.drop('word_freq_650', axis = 1)
Dtest = Dtest.drop('word_freq_lab', axis = 1)
Dtest = Dtest.drop('word_freq_labs', axis = 1)
Dtest = Dtest.drop('word_freq_telnet', axis = 1)
Dtest = Dtest.drop('word_freq_85', axis = 1)
Dtest = Dtest.drop('word_freq_1999', axis = 1)
Dtest = Dtest.drop('word_freq_edu', axis = 1)
NB = BernoulliNB()
X = df.drop('ham', axis = 1)
X = X.drop('Id', axis = 1)
scores = cross_val_score(NB, D, Y, cv=100, scoring = score)
sum(scores)/len(scores)
NB.fit(D ,Y)
testNB = NB.predict(D)
testPredNB = NB.predict(Dtest) 
confusion_matrix(testNB, Y)
proba = cross_val_predict(NB, D, Y, cv=10, method = 'predict_proba')
fpr, tpr, threshold = roc_curve(Y, proba[:,1])
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
threshold[558]
di = 999
for i in range(len(threshold)):
    d = math.sqrt(((fpr[i])**2) + ((tpr[i]-1)**2))
    if (d < di):
        j = i
        di = d
    
j
NB = BernoulliNB(class_prior = [1-threshold[j], threshold[j]])
scores = cross_val_score(NB, D, Y, cv=100, scoring = score)
sum(scores)/len(scores)
NB.fit(D ,Y)
testPredNB = NB.predict(Dtest) 
knn = KNeighborsClassifier(n_neighbors= 5)
scores = cross_val_score(knn, D, Y, cv=10, scoring = score)
sum(scores)/len(scores)
knn.fit(D,Y)
testPredKNN = knn.predict(Dtest)
testKNN = knn.predict(D)
confusion_matrix(testKNN, df.ham)
proba = cross_val_predict(knn, D, Y, cv=10, method = 'predict_proba')
fpr, tpr, threshold = roc_curve(Y, proba[:,1])
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
#Vote = VotingClassifier(estimators=[('NB',NB), ('knn', knn)])
#scores = cross_val_score(Vote, D, Y, cv=10, scoring = score)
#sum(scores)/len(scores)
#Vote.fit(D,Y)
#testVote = Vote.predict(D)
#confusion_matrix(testVote, df.ham)
#testPredVote = Vote.predict(Dtest)
#testPredVote = testPredVote > 0
    
testPred = []
for i in range(len(testPredKNN)):
    if(testPredKNN[i] == False and testPredNB[i] == False):
        testPred.append(False)
    else:
        testPred.append(True)

testPredNB[0]
testPred = pd.DataFrame(testPred)
w = pd.DataFrame(sid)
w.columns = ["id"]
w["ham"] = testPred
w
w.to_csv("resposta.csv", index = False)

