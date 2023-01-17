# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import os



!ls ../input



dataset='benchmark'

feature=['pssm','w2v']

totality_features=[]

labels=[]



labels=np.load("../input/sublocalisation-of-protien/benchmark_pssm_label_multi.npy")

pssm_features=np.load("../input/sublocalisation-of-protien/benchmark_pssm_feat_multi.npy")

totality_features=np.load("../input/sublocalisation-of-protien/benchmark_total_feat_multi.npy")

x_train=totality_features
X=np.load('../input/label-5/benchmark_label5_GA.npy')

label=labels[:,4]

label.shape


import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



from sklearn.svm import SVC

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

a = SVC(kernel='rbf', C=4,gamma='auto')

import numpy as np

from sklearn.metrics import accuracy_score

acc_svm = []

acc_svm_training=[]

for train, test in kf.split(X, label):

    X_train, X_test = X[train], X[test]

    Y_train, Y_test = label[train], label[test]

    

    a.fit(X_train, Y_train)

    pred = a.predict(X_test)

    pred1=a.predict(X_train)

    acc_svm_training.append(accuracy_score(pred1,Y_train))

    acc_svm.append(accuracy_score(pred, Y_test))



print ('\nFold Accuracies: ', acc_svm)

print ('\nAccuracy SVM CF: ', np.mean(acc_svm))
print ('\nFold Accuracies: ', acc_svm_training)

print ('\nAccuracy SVM CF: ', np.mean(acc_svm_training))