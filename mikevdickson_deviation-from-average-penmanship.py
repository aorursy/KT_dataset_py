# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import scipy as sp

import matplotlib.pyplot as plt 

%matplotlib inline

from sklearn.datasets import load_digits



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



from sklearn import svm

from sklearn import neighbors

from sklearn import naive_bayes

from sklearn import ensemble



from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv').values

Xte = pd.read_csv('../input/test.csv').values

ytr = train[:,0]

Xtr = train[:,1:]
print(Xte.shape)

print(Xtr.shape)

print(ytr.shape)
RES=28

n_classes = 10

rand_seed = 99
# plot a single digit

plt.gray()

plt.matshow(Xtr[2,:].reshape(RES,RES))
def get_average(X,y):

    X_avg = np.zeros((n_classes,RES*RES))

    for dig in range(n_classes):

        mask = [i for i in range(len(y)) if y[i]==dig]

        avg = np.average(X[mask,:],axis=0)

        X_avg[dig] = avg

    return X_avg
def make_features(X,avg):

    Xf = np.zeros((len(X),n_classes))

    for dig in range(n_classes):

        delta = X-avg[dig]

        Xf[:,dig] = np.var(delta,axis=1)

    return Xf
def KFoldPrep(X,y):

    global kf, kf_Xtr_feat, kf_Xte_feat

    kf = KFold(n_splits=3, shuffle=True, random_state=rand_seed)

    kf_Xtr_feat = []

    kf_Xte_feat = []

    for itr, ite in kf.split(X):

        avg = get_average(X[itr],y[itr])

        kf_Xtr_feat.append(make_features(X[itr],avg))

        kf_Xte_feat.append(make_features(X[ite],avg))



def score(X,y,clf):

    s = []

    for i,(itr,ite) in enumerate(kf.split(X)):

        clf.fit(kf_Xtr_feat[i],y[itr])

        pred = clf.predict(kf_Xte_feat[i])

        s.append(accuracy_score(y[ite],pred))

        print('{0:0.5f}'.format(s[-1]))

    s_avg = np.mean(s)

    print('-> {0:0.7f}'.format(s_avg))

    return s_avg
names = [#'Linear SVM',

         'Nearest Neighbor',

         'Naive Bayes',

         'Extra Trees',

         'Random Forest'

        ]

clfs = [#svm.SVC(kernel='linear',probability=True),

        neighbors.KNeighborsClassifier(5),

        naive_bayes.GaussianNB(),

        ensemble.ExtraTreesClassifier(n_estimators=50),

        ensemble.RandomForestClassifier(n_estimators=50)

       ]
KFoldPrep(Xtr,ytr)
scores = []

for i,model in enumerate(clfs):

    print('Fitting {}:'.format(names[i]))

    scores.append(score(Xtr,ytr,model))
Xtr_avg = get_average(Xtr,ytr)

Xtr_feat = make_features(Xtr,Xtr_avg)

Xte_feat = make_features(Xte,Xtr_avg)



pred = np.zeros((len(Xte),n_classes))

for model in clfs:

    model.fit(Xtr_feat,ytr)

    pred += model.predict_proba(Xte_feat)

    

pred = np.argmax(pred,axis=1)
print(Xte.shape)

print(pred.shape)
submission = pd.DataFrame(pred).reset_index()

submission.columns = ['ImageId','Label']

submission['ImageId'] = submission['ImageId']+1

submission.to_csv('submission.csv',index=False)