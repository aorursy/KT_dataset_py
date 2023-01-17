import pandas as pd

import csv

import pickle

import numpy as np

from sklearn import decomposition

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import numpy as np

from sklearn.model_selection import train_test_split

import keras

from keras.callbacks import LambdaCallback

from keras.layers import Conv1D, Flatten

from keras.layers import Dense ,Dropout,BatchNormalization

from keras.models import Sequential 

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical 

from keras import regularizers

from sklearn import preprocessing

from sklearn.ensemble import  VotingClassifier

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import metrics

from sklearn import ensemble

from sklearn import gaussian_process

from sklearn import linear_model

from sklearn import naive_bayes

from sklearn import neighbors

from sklearn import svm

from sklearn import tree

from sklearn import discriminant_analysis

from sklearn import model_selection

from xgboost.sklearn import XGBClassifier 

import os

print(os.listdir("../input"))

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
neg = pd.read_table("../input/Neg_DPC.tsv")

neg.shape

neg=neg.rename(index=str, columns={"#": "label"})

neg["label"]=0



pos = pd.read_table("../input/Pos_DPC.tsv")

pos.shape

pos=pos.rename(index=str, columns={"#": "label"})

pos["label"]=1







frames = [pos,neg]

df=pd.concat(frames)



X=df.drop("label",axis=1)

y=df["label"]



X=X.values

y=y.values



df.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]









#create table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy' ]

MLA_compare = pd.DataFrame(columns = MLA_columns)







#index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

   # cv_results = model_selection.cross_validate(alg, X_train, y_train)

    alg.fit(X_train, y_train)

    y_pred=alg.predict(X_test)

    score=metrics.accuracy_score(y_test, y_pred)

    

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] =score



    

    

    row_index+=1



    





#MLA_predict
MLA_compare
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif



# Create an SelectKBest object to select features with two best ANOVA F-Values

fvalue_selector = SelectKBest(f_classif, k=52)



# Apply the SelectKBest object to the features and target

X_kbest = fvalue_selector.fit_transform(X, y)



print('Original number of features:', X.shape[1])

print('Reduced number of features:', X_kbest.shape[1])
X_train, X_test, y_train, y_test = train_test_split( X_kbest, y, test_size=.3)

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]









#create table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy' ]

MLA_compare = pd.DataFrame(columns = MLA_columns)







#index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

   # cv_results = model_selection.cross_validate(alg, X_train, y_train)

    alg.fit(X_train, y_train)

    y_pred=alg.predict(X_test)

    score=metrics.accuracy_score(y_test, y_pred)

    

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] =score



    

    

    row_index+=1



    





#MLA_predict
MLA_compare


kfold = StratifiedKFold(n_splits=100, shuffle=True)

cvscores = []



iterator = 1

cv_score = 0

for train, test in kfold.split(X_kbest, y):

    print('Fold : '+str(iterator))

    # giving 100% accuracy on n_estimator 50

    RandomForest = discriminant_analysis.LinearDiscriminantAnalysis().fit(X_kbest[train],y[train].ravel())

    pred = np.round(RandomForest.predict(X_kbest[test]))

    tn, fp, fn, tp = confusion_matrix(y[test], pred, labels=[1,0]).ravel()

    acc = np.round(((tn+tp)/(tn+fp+fn+tp))*100, 2)

    cvscores.append([tn,fp,fn,tp,acc])

    iterator=iterator+1

    print([tn,fp,fn,tp,acc])

    cv_score = cv_score + acc
print('\n\rFinal JackKnife Score = ', np.round(cv_score/kfold.n_splits,2))
X_train, X_test, y_train, y_test = train_test_split( X_kbest, y, test_size=.3)

clf=discriminant_analysis.LinearDiscriminantAnalysis().fit(X_train,y_train)

bstlfy=ensemble.BaggingClassifier(base_estimator=clf,n_estimators=20)

bstlfy=bstlfy.fit(X_train,y_train)

prediction=bstlfy.predict(X_test)

    



true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_test, prediction).ravel()
print("true_negative: ",true_negative)

print("false_positive: ",false_positive)

print("false_negative: ",false_negative)

print("true_positive: ",true_positive)

print("\n\n Accuracy Measures\n\n")



Accuracy=(true_positive+true_negative)/(true_positive+false_positive+true_negative+false_negative)

print("Accuracy: ",Accuracy)



Sensitivity=true_positive/(true_positive+false_negative)

print("Sensitivity: ",Sensitivity)



False_Positive_Rate=false_positive/(false_positive+true_negative)

print("False_Positive_Rate: ",False_Positive_Rate)



Specificity=true_negative/(false_positive + true_negative)

print("Specificity: ",Specificity)



        #FDR à 0 means that very few of our predictions are wrong

False_Discovery_Rate=false_positive/(false_positive+true_positive)

print("False_Discovery_Rate: ",False_Discovery_Rate)



Positive_Predictive_Value =true_positive/(true_positive+false_positive)

print("Positive_Predictive_Value: ",Positive_Predictive_Value)


kfold = StratifiedKFold(n_splits=100, shuffle=True)

cvscores = []



iterator = 1

cv_score = 0

for train, test in kfold.split(X_kbest, y):

    print('Fold : '+str(iterator))

    # giving 100% accuracy on n_estimator 50

    clf=ensemble.GradientBoostingClassifier().fit(X_train,y_train)

    bstlfy=ensemble.BaggingClassifier(base_estimator=clf,n_estimators=20)

    bstlfy=bstlfy.fit(X_train,y_train)

    pred = np.round(bstlfy.predict(X_kbest[test]))

    tn, fp, fn, tp = confusion_matrix(y[test], pred, labels=[1,0]).ravel()

    acc = np.round(((tn+tp)/(tn+fp+fn+tp))*100, 2)

    cvscores.append([tn,fp,fn,tp,acc])

    iterator=iterator+1

    print([tn,fp,fn,tp,acc])

    cv_score = cv_score + acc
print('\n\rFinal JackKnife Score = ', np.round(cv_score/kfold.n_splits,2))
prediction=bstlfy.predict(X_test)

true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_test, prediction).ravel()

print("true_negative: ",true_negative)

print("false_positive: ",false_positive)

print("false_negative: ",false_negative)

print("true_positive: ",true_positive)

print("\n\n Accuracy Measures\n\n")



Accuracy=(true_positive+true_negative)/(true_positive+false_positive+true_negative+false_negative)

print("Accuracy: ",Accuracy)



Sensitivity=true_positive/(true_positive+false_negative)

print("Sensitivity: ",Sensitivity)



False_Positive_Rate=false_positive/(false_positive+true_negative)

print("False_Positive_Rate: ",False_Positive_Rate)



Specificity=true_negative/(false_positive + true_negative)

print("Specificity: ",Specificity)



        #FDR à 0 means that very few of our predictions are wrong

False_Discovery_Rate=false_positive/(false_positive+true_positive)

print("False_Discovery_Rate: ",False_Discovery_Rate)



Positive_Predictive_Value =true_positive/(true_positive+false_positive)

print("Positive_Predictive_Value: ",Positive_Predictive_Value)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot

probs=bstlfy.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, probs)



# calculate AUC

auc = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc)
pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

pyplot.plot(fpr, tpr, marker='.')

# show the plot

pyplot.show()
filename = 'finalized_model.sav'

pickle.dump(bstlfy, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.score(X_test, y_test)

print(result)
a=bstlfy.predict(X_train[1:])

a.shape
X_train[:1]

y_train[:1]