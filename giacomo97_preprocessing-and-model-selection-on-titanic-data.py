# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/titanic/train.csv")

data.sample(frac=1,random_state=69)

data.head()
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

import matplotlib.pyplot as plt



def evaluate(y_test, y_pred):

    return {

        "accuracy" : accuracy_score(y_test, y_pred),

        #"precision" : precision_score(y_test, y_pred),

        "recall" : recall_score(y_test, y_pred),

        "AUC" : roc_auc_score(y_test, y_pred),

        "f1-measure" :  f1_score(y_test, y_pred, average='micro')

    }

def sum_metrics(a,b):

    for key, value in b.items():

        if key not in a.keys():

            a[key]=value

        else:

            a[key]=a[key]+value

    return a



def divide_all_metrics(a,scalar):

    for key, value in a.items():

        a[key]=value/scalar

    return a



def metricsKeeper(listOfMetrics,newMetrics,i):

    for key,value in newMetrics.items():

        if key not in listOfMetrics.keys():

            listOfMetrics[key]=[]

        listOfMetrics[key].insert(i,value)

    return listOfMetrics



def plotMetrics(metrics):

    plt.figure()

    for key, value in metrics.items():

        plt.plot(range(0,len(value)),value,label=key)

        plt.legend()

    plt.show()
X=data[data.columns.difference(['PassengerId',"Name","Survived"])]

y=data["Survived"]
continuos_features=["Age","Fare"]

categorical_features=["Pclass","Sex","SibSp","Parch","Ticket","Cabin"]
from sklearn.impute import SimpleImputer



f0,f1=0,1

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

X_imputed=imp_mean.fit_transform(X[continuos_features])
plt.scatter(X_imputed[:,f0],X_imputed[:,f1],c=y)

plt.show()
from sklearn.preprocessing import Normalizer



norm_enc = Normalizer()

X_normalized=norm_enc.fit_transform(X_imputed)

plt.scatter(X_normalized[:,f0],X_normalized[:,f1],c=y)

plt.show()
from sklearn.preprocessing import StandardScaler

scaler_enc = StandardScaler()



X_normalized=scaler_enc.fit_transform(X_imputed)

plt.scatter(X_normalized[:,f0],X_normalized[:,f1],c=y)

plt.show()
from sklearn.preprocessing import MinMaxScaler



min_max_enc = MinMaxScaler()



X_normalized=min_max_enc.fit_transform(X_imputed)

plt.scatter(X_normalized[:,f0],X_normalized[:,f1],c=y)

plt.show()
from sklearn.preprocessing import Binarizer



binarizer_enc = Binarizer()



X_normalized=binarizer_enc.fit_transform(X_imputed)

plt.scatter(X_normalized[:,f0],X_normalized[:,f1],c=y)

plt.show()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import OneHotEncoder



class Preprocesser:

    def __init__(self,continuos_features, categorical_features):

        self.continuos_features=continuos_features

        self.categorical_features=categorical_features

        

        self.imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

        self.imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

        

        self.one_hot_enc = OneHotEncoder(handle_unknown='ignore')

        self.norm_enc = Normalizer()



    def fit(self,X_train):

        self.imp_mean.fit(X_train[self.continuos_features])

        self.imp_most_frequent.fit(X_train[self.categorical_features])

        

        train_cat_feat_imputing=self.imp_most_frequent.transform(X_train[self.categorical_features])

        self.one_hot_enc.fit(train_cat_feat_imputing)



        train_cont_feat_imputing=self.imp_mean.transform(X_train[self.continuos_features])

        self.norm_enc.fit(train_cont_feat_imputing)



    def transform(self,X):

        X_cat_imputed=self.imp_most_frequent.transform(X[self.categorical_features])

        X_cont_imputed=self.imp_mean.transform(X[self.continuos_features])

        

        X_cat_encoded=self.one_hot_enc.transform(X_cat_imputed).toarray()

        #X_cont_encoded=self.norm_enc.transform(X_cont_imputed)

        X_cont_encoded=X_cont_imputed



        return np.concatenate((X_cat_encoded, X_cont_encoded), axis=1)
import itertools



iperpar_values={

    "C"      : [2**i for i in range(-1,1)],

    "kernel" : ["poly","rbf"],

    "degree" : [2+i for i in range(0,2)],

    "gamma"  : [10**i for i in range(-1, 1)],

    "coef0"   : [0+i for i in np.linspace(-1,1,2)]

}

keys = iperpar_values.keys()

values = (iperpar_values[key] for key in keys)

combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
import sys

from sklearn.model_selection import train_test_split

from sklearn import svm



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)

preprocesser=Preprocesser(continuos_features,categorical_features)

preprocesser.fit(X_train)

X_train=preprocesser.transform(X_train)

X_test=preprocesser.transform(X_test)



best_metrics={}

best_combination={}

metricsList={}

j=0

for comb in combinations:

    sys.stdout.write("\r Evaluating model %i out of %i" % (j+1 ,len(combinations)))

    svm_clf=svm.SVC(gamma="auto")

    svm_clf.fit(X_train,y_train)

    predictions=svm_clf.predict(X_test)

    metrics=evaluate(y_test,predictions)

    metricsList=metricsKeeper(metricsList,metrics,j)

    if (("f1-measure" not in best_metrics.keys()) or metrics["f1-measure"]>best_metrics["f1-measure"]):

        best_metrics=metrics

        best_combination=comb

    j=j+1

plotMetrics(metricsList)

print("")

print("The best combination was:")

print(best_combination)

print("Which had the following results:")

print(best_metrics)
import math 

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



n_folds=3



n_examples=X.shape[0]

best_metrics={}

best_combination={}

metricsList={}

j=0



for comb in combinations:

    sys.stdout.write("\r Evaluating model %i out of %i" % (j+1 ,len(combinations)))

    avg_metrics={}

    svm_clf=svm.SVC(C=comb["C"],kernel=comb["kernel"],degree=comb["degree"],gamma=comb["gamma"],coef0=comb["coef0"])

    for i in range(0,n_folds):

        fold_start=math.trunc(n_examples/n_folds)*i

        fold_end=math.trunc(n_examples/n_folds)*(i+1)



        X_train=X.iloc[np.r_[0:fold_start,fold_end:n_examples],:]

        y_train=y.iloc[np.r_[0:fold_start,fold_end:n_examples]]

        X_test=X.iloc[fold_start:fold_end,:]

        y_test=y.iloc[fold_start:fold_end]

        

        preprocesser.fit(X_train)

        X_train=preprocesser.transform(X_train)

        X_test=preprocesser.transform(X_test)

        

        svm_clf.fit(X_train,y_train)

        predictions=svm_clf.predict(X_test)

        metrics=evaluate(y_test,predictions)

        sum_metrics(avg_metrics,metrics)

    avg_metrics=divide_all_metrics(avg_metrics,n_folds)

    metricsList=metricsKeeper(metricsList,avg_metrics,j)

    if (("f1-measure" not in best_metrics.keys()) or avg_metrics["f1-measure"]>best_metrics["f1-measure"]):

        best_metrics=avg_metrics

        best_combination=comb

    j=j+1

plotMetrics(metricsList)

print("")

print("The best combination was:")

print(best_combination)

print("Which had the following results:")

print(best_metrics)