import pandas as pd

import numpy as np

import os



!ls ../input



dataset='benchmark'

feature=['pssm','w2v']

totality_features=[]

labels=[]



labels=np.load("../input/benchmark_pssm_label_multi.npy")

pssm_features=np.load("../input/benchmark_pssm_feat_multi.npy")



w2v_features=np.load("../input/benchmark_w2v_feat_multi.npy")

for i in range(len(w2v_features)):

    totality_features.append(np.concatenate((pssm_features[i],w2v_features[i])))

totality_features=np.array(totality_features)

totality_features.shape

x_train=totality_features
labels.shape
x_train.shape
label3=labels[:,3]
#SVM class label1

import numpy as np

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB

from genetic_selection import GeneticSelectionCV

from sklearn.svm import SVC



x_train.astype(dtype='object')



label3.astype(dtype='object')





estimator=SVC(kernel='rbf', C=1,gamma='auto')

selector = GeneticSelectionCV(estimator,cv=5,verbose=1,scoring="accuracy",max_features=1200,n_population=100,crossover_proba=0.3,mutation_proba=0.1,n_generations=60,crossover_independent_proba=0.1,mutation_independent_proba=0.05,tournament_size=3,caching=True,n_jobs=-1)

selector = selector.fit(x_train,label3)

x_train_label1=selector.transform(x_train)

x_train_label1.shape
np.save('benchmark_label4_GA.npy',x_train_label1)
np.save('benchmark_label4_features_GA.npy',selector.support_ )