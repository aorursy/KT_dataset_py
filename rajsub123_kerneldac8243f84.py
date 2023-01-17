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
#importing libraries



import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#reading the file

#os.chdir(r'''E:\Python ML Study\predicting-a-pulsar-star''')

df= pd.read_csv("../input/pulsar_stars.csv")
df.head()
df.info()
df.describe()
df.target_class.dtype
df = df.astype({"target_class": "category"})
df.info()
from sklearn.utils import resample

# Separate majority and minority classes

df_majority = df[df.target_class==0]

df_minority = df[df.target_class==1]

 

# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,  

                                 n_samples=16259,    

                                 random_state=123) 

 

# Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

 

# Display new class counts

#df_upsampled.balance.value_counts()
df_upsampled.describe()
df_upsampled.columns
X= df_upsampled[[' Mean of the integrated profile',

       ' Standard deviation of the integrated profile',

       ' Excess kurtosis of the integrated profile',

       ' Skewness of the integrated profile', ' Mean of the DM-SNR curve',

       ' Standard deviation of the DM-SNR curve',

       ' Excess kurtosis of the DM-SNR curve', ' Skewness of the DM-SNR curve']]



Y= df_upsampled['target_class']
from sklearn.model_selection import train_test_split

from sklearn import metrics

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
X.shape
Y.shape
df_upsampled.target_class.value_counts()
#for building a decison tree

from sklearn import tree

clf_tree= tree.DecisionTreeClassifier()

clf_tree= clf_tree.fit(X_train,Y_train)
# to predict and get prediction metrics 

Y_predict_tree = clf_tree.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(Y_test, Y_predict_tree)





from sklearn.metrics import confusion_matrix

tn_tree, fp_tree, fn_tree, tp_tree = confusion_matrix(Y_test, Y_predict_tree, labels=None, sample_weight=None).ravel()

print(tn_tree, fp_tree, fn_tree, tp_tree)
# to have a look at the developed tree

import graphviz 

dot_data = tree.export_graphviz(clf_tree, out_file=None) 

graph = graphviz.Source(dot_data)

graph.render("star")

graph
# for building a random forest classifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

clf_rforest = RandomForestClassifier(n_estimators=100, max_depth=2,

                            random_state=0)

clf_rforest.fit(X_train,Y_train)
# to predict and get prediction metrics 

Y_predict_rforest = clf_rforest.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(Y_test, Y_predict_rforest)





tn_rforest, fp_rforest, fn_rforest, tp_rforest = confusion_matrix(Y_test, Y_predict_rforest, labels=None, sample_weight=None).ravel()

print(tn_rforest, fp_rforest, fn_rforest, tp_rforest)
# building a logistic regression model

from sklearn.linear_model import LogisticRegression



clf_log = LogisticRegression(random_state=0, solver='lbfgs',

                          multi_class='multinomial').fit(X_train, Y_train)
# to predict and get prediction metrics 

Y_predict_log = clf_log.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(Y_test, Y_predict_log)



from sklearn.metrics import confusion_matrix

tn_log, fp_log, fn_log, tp_log = confusion_matrix(Y_test, Y_predict_log, labels=None, sample_weight=None).ravel()

print(tn_log, fp_log, fn_log, tp_log)
# to build a svm 

from sklearn import svm

clf_svm = svm.SVC(gamma=0.001)

clf_svm.fit(X_train, Y_train)  
# to predict and get prediction metrics 

Y_predict_svm = clf_svm.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(Y_test, Y_predict_svm)





from sklearn.metrics import confusion_matrix

tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(Y_test, Y_predict_svm, labels=None, sample_weight=None).ravel()

print(tn_svm, fp_svm, fn_svm, tp_svm)
#adaptive boosting model

from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

clf_aboost = AdaBoostClassifier(n_estimators=100, random_state=0)

clf_aboost.fit(X_train, Y_train)
# to predict and get prediction metrics 

Y_predict_aboost = clf_aboost.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(Y_test, Y_predict_aboost)





from sklearn.metrics import confusion_matrix

tn_aboost, fp_aboost, fn_aboost, tp_aboost = confusion_matrix(Y_test, Y_predict_aboost, labels=None, sample_weight=None).ravel()

print(tn_aboost, fp_aboost, fn_aboost, tp_aboost)
#KNN model

from sklearn.neighbors import KNeighborsClassifier

neigh_knn = KNeighborsClassifier(n_neighbors=3)

neigh_knn.fit(X_train, Y_train)
# to predict and get prediction metrics

Y_predict_knn = neigh_knn.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(Y_test, Y_predict_knn)



from sklearn.metrics import confusion_matrix

tn_knn, fp_knn, fn_knn, tp_knn = confusion_matrix(Y_test, Y_predict_knn, labels=None, sample_weight=None).ravel()

print(tn_knn, fp_knn, fn_knn, tp_knn)
#Naive Base Model

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

nb_model.fit(X_train,Y_train)

# to predict and get prediction metrics

Y_predict_nb = nb_model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(Y_test, Y_predict_nb)



from sklearn.metrics import confusion_matrix

tn_nb, fp_nb, fn_nb, tp_nb = confusion_matrix(Y_test, Y_predict_nb, labels=None, sample_weight=None).ravel()

print(tn_nb, fp_nb, fn_nb, tp_nb)