#https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import Perceptron

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import *

import pandas as pd

from pandas.tools.plotting import scatter_matrix

import random

import itertools

import seaborn as sns



sns.set(style = 'darkgrid')

% matplotlib inline
bc = pd.read_csv("../input/data.csv")
bc.head(1)
bcs = pd.DataFrame(preprocessing.scale(bc.ix[:,2:32]))

bcs.columns = list(bc.ix[:,2:32].columns)

bcs['diagnosis'] = bc['diagnosis']
corr = bcs.corr()

fg, ax = plt.subplots(figsize = (11,9))



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(corr, mask = mask, linewidths = .5, square = True);
sns.lmplot(x = 'fractal_dimension_mean', y = 'radius_mean',  hue='diagnosis', size = 4, data = bcs);
p = sns.PairGrid(bcs.ix[:,20:32], hue = 'diagnosis', palette = 'Reds')

p.map_upper(plt.scatter, s = 20, edgecolor = 'w')

p.map_diag(plt.hist)

p.map_lower(sns.kdeplot, cmap = 'GnBu_d')

p.add_legend()



p.figsize = (30,30)
mbc = pd.melt(bcs, "diagnosis", var_name="measurement")

fig, ax = plt.subplots(figsize=(10,5))

p = sns.violinplot(ax=ax, x="measurement", y="value", hue="diagnosis", split = True, data=mbc, inner = 'quartile', palette = 'Set2');

p.set_xticklabels(rotation = 90, labels = list(bcs.columns));
sns.swarmplot(x = 'diagnosis', y = 'concave points_worst',palette = 'Set2', data = bc);
sns.jointplot(x = bc['concave points_worst'], y = bc['area_mean'],kind='reg', color="#4CB391", size = 6);
X = bcs.ix[:,0:30]



y = bcs['diagnosis']

class_names = list(y.unique())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
m_eval = pd.DataFrame(columns = ['method','trainscore','testscore','True Positive','True Negative'])
def addeval(method, train, test, tpos, tneg):

    global m_eval

    d = pd.DataFrame([[method, train, test, tpos, tneg]],columns = ['method','trainscore','testscore','True Positive','True Negative'])

    m_eval = m_eval.append(d)
svc = SVC(kernel = 'linear',C=.1, gamma=10, probability = True)

svc.fit(X,y)

svc_pred = svc.fit(X_train, y_train).predict(X_test)

#t = pd.DataFrame(svc.predict_proba(X_test))

print(svc.score(X_train,y_train), svc.score(X_test, y_test))



mtrx = confusion_matrix(y_test,svc_pred)

mtrx
addeval('SVM',svc.score(X_train,y_train), svc.score(X_test, y_test),mtrx[1,1],mtrx[0,0])
###LogReg

lr = LogisticRegression(penalty = 'l2', dual = True)

lr_pred = lr.fit(X_train, y_train).predict(X_test)

print(lr.score(X_train,y_train), lr.score(X_test, y_test),lr.score(X_train,y_train)-lr.score(X_test,y_test))



mtrx = confusion_matrix(y_test,lr_pred)

mtrx
addeval('Log Reg',lr.score(X_train,y_train), lr.score(X_test, y_test),mtrx[1,1],mtrx[0,0])
###Neural Net

nn = MLPClassifier(solver='adam', activation = 'logistic',hidden_layer_sizes=(10, 50), random_state=1)



nn_pred = nn.fit(X_train,y_train).predict(X_test)

print(nn.score(X_train,y_train), nn.score(X_test,y_test), nn.score(X_train,y_train)-nn.score(X_test,y_test))



mtrx = confusion_matrix(y_test,nn_pred)

mtrx
addeval('Neural Net',nn.score(X_train,y_train), nn.score(X_test, y_test),mtrx[1,1],mtrx[0,0])
#Gauss Naive Bayes

gauss = GaussianNB()

gauss_pred = gauss.fit(X_train, y_train).predict(X_test)

print(gauss.score(X_train,y_train), gauss.score(X_test,y_test), gauss.score(X_train,y_train)-gauss.score(X_test,y_test))



mtrx = confusion_matrix(y_test,gauss_pred)

mtrx
addeval('Gauss NB',gauss.score(X_train,y_train), gauss.score(X_test, y_test),mtrx[1,1],mtrx[0,0])
#Perceptron

#Perceptron and SGDClassifier share the same underlying implementation. In fact, Perceptron() is 

#equivalent to SGDClassifier(loss=”perceptron”, eta0=1, learning_rate=”constant”, penalty=None).

perc = Perceptron(alpha = 1, penalty = None,fit_intercept = False)

perc_pred = perc.fit(X_train, y_train).predict(X_test)

print(perc.score(X_train,y_train), perc.score(X_test,y_test), perc.score(X_train,y_train)-perc.score(X_test,y_test))



mtrx = confusion_matrix(y_test,perc_pred)

mtrx
addeval('Perceptron',perc.score(X_train,y_train), perc.score(X_test, y_test),mtrx[1,1],mtrx[0,0])
#Ensemble

ens = VotingClassifier(estimators=[('LR', lr), ('SVC', svc), ('NN', nn)], 

                       voting='soft', weights=[1,2,3])

ens_pred = ens.fit(X_train, y_train).predict(X_test)

print(ens.score(X_train,y_train), ens.score(X_test,y_test), ens.score(X_train,y_train)-ens.score(X_test,y_test))



mtrx = confusion_matrix(y_test,ens_pred)

mtrx
addeval('Ensemble',ens.score(X_train,y_train), ens.score(X_test, y_test),mtrx[1,1],mtrx[0,0])
m_eval
mm1_eval = pd.melt(m_eval[['method','True Positive','True Negative']], "method", var_name="Measurement")

mm2_eval = pd.melt(m_eval[['method','trainscore','testscore']], "method", var_name="Measurement")
p = sns.pointplot(x="method", y="value", hue="Measurement", data=mm1_eval)

labs = list(m_eval['method'])

p.set_xticklabels(labs, rotation=90);
p = sns.pointplot(x="method", y="value", hue="Measurement", data=mm2_eval)

labs = list(m_eval['method'])

p.set_xticklabels(labs, rotation=90);