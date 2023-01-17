import pandas as pd

import csv

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
# HBP

data = pd.read_csv('../input/protein-sequences/data 4/hbp.txt', sep=">",header=None)

sequences=data[0].dropna()

labels=data[1].dropna()

sequences.reset_index(drop=True, inplace=True)

labels.reset_index(drop=True, inplace=True)

list_of_series=[sequences.rename("sequences"),labels.rename("Name")]

df_hbp = pd.concat(list_of_series, axis=1)

df_hbp['label']='hbp'

df_hbp.head()
# not HBP

data = pd.read_csv('../input/protein-sequences/data 4/non-hbp.txt', sep=">",header=None)

sequences=data[0].dropna()

labels=data[1].dropna()

sequences.reset_index(drop=True, inplace=True)

labels.reset_index(drop=True, inplace=True)

list_of_series=[sequences.rename("sequences"),labels.rename("Name")]

df_N_hbp = pd.concat(list_of_series, axis=1)

df_N_hbp['label']='non-hbp'

df_N_hbp.head()
frames = [df_hbp,df_N_hbp]

df=pd.concat(frames)

df.head()
arr=[]

for i in df.sequences:

    arr.append(len(i))

    

arr=np.asarray(arr)

print("Minimum length of string is = ",(arr.min()))

minlength=arr.min()

from keras.preprocessing import text, sequence

from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split



# maximum length of sequence, everything afterwards is discarded!

max_length = minlength



#create and fit tokenizer

tokenizer = Tokenizer(char_level=True)

tokenizer.fit_on_texts(df.sequences)

#represent input data as word rank number sequences

X = tokenizer.texts_to_sequences(df.sequences)

X = sequence.pad_sequences(X, maxlen=max_length)
from sklearn.preprocessing import LabelBinarizer

import keras



lb = LabelBinarizer()

Y = lb.fit_transform(df.label)

print(len(X[0]))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=0)



pca = decomposition.PCA(n_components=2)

pca.fit(X_train)

X_train = pca.transform(X_train)



pca = decomposition.PCA(n_components=2)

pca.fit(X_test)

X_test = pca.transform(X_test)



kfold = StratifiedKFold(n_splits=10, shuffle=True)

cvscores = []



iterator = 1

cv_score = 0

for train, test in kfold.split(X_train, y_train):

    print('Fold : '+str(iterator))

    # giving 100% accuracy on n_estimator 50

    RandomForest = RandomForestClassifier(n_estimators=10, oob_score=True, n_jobs=-1, warm_start=True).fit(X_train[train],y_train[train].ravel())

    pred = np.round(RandomForest.predict(X_train[test]))

    tn, fp, fn, tp = confusion_matrix(y_train[test], pred, labels=[1,0]).ravel()

    acc = np.round(((tn+tp)/(tn+fp+fn+tp))*100, 2)

    cvscores.append([tn,fp,fn,tp,acc])

    iterator=iterator+1

    print([tn,fp,fn,tp,acc])

    cv_score = cv_score + acc

print('\n\rFinal 10CV Score = ', np.round(cv_score/kfold.n_splits,2),'\n\rResults are Saved in CrossValidationResults.csv\n\r')

def frequencyVec(seq):

    encoder = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',

               'Y']

    fv = [0 for x in range(20)]

    i = 0

    for i in range(20):

        fv[i - 1] = seq.count(encoder[i])

    return fv

X_frequencyVec=[]

for i in df.sequences:

    X_frequencyVec.append(frequencyVec(i))

                      

                      

X_frequencyVec = np.asarray(X_frequencyVec)

X_frequencyVec.shape

                    
X_train, X_test, y_train, y_test = train_test_split(X_frequencyVec, df.label, test_size=.3, random_state=0)

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
def AAPIV(seq):

    encoder = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',

               'Y']

    apv = [0 for x in range(20)]

    i = 1

    sum = 0

    for i in range(20):

        j = 0

        for j in range(len(seq)):

            if seq[j] == encoder[i]:

                sum = sum + j + 1

        apv[i] = sum

        sum = 0

    return apv[1:] + apv[0:1]

X_AAPIV=[]



for i in df.sequences:

    X_AAPIV.append(AAPIV(i))

                      

                      

X_AAPIV = np.asarray(X_AAPIV)

X_AAPIV.shape

X_train, X_test, y_train, y_test = train_test_split(X_AAPIV , df.label, test_size=.3, random_state=0)

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
def PRIM(seq):

    encoder = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',

               'Y']

    prim = [[0 for x in range(20)] for y in range(20)]

    i = 0

    for i in range(20):

        aa1 = encoder[i]

        aa1index = -1

        for x in range(len(seq)):

            if seq[x] == aa1:

                aa1index = x + 1

                break

        if aa1index != -1:

            j = 0

            for j in range(20):

                if j != i:

                    aa2 = encoder[j]

                    aa2index = 0

                    for y in range(len(seq)):

                        if seq[y] == aa2:

                            aa2index = aa2index + ((y + 1) - aa1index)

                    prim[i][j] = int(aa2index)

    return prim

X_PRIM=[]



for i in df.sequences:

    X_PRIM.append(PRIM(i))

                      

                      

X_PRIM = np.asarray(X_PRIM)

X_PRIM.shape
X_PRIM=X_PRIM.reshape(246, 20*20)

X_PRIM.shape
X_train, X_test, y_train, y_test = train_test_split(X_PRIM, df.label, test_size=.3, random_state=0)

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
from sklearn.model_selection import RandomizedSearchCV

from sklearn import tree





clfETC=tree.ExtraTreeClassifier()





param_dist = {"max_depth": [30,50,23,25,17,None],

               "max_features": [0.03,0.07,0.10,0.15,0.20,0.30,"auto"],

              

              "max_depth":[20,40,50,100],

              

              "min_samples_leaf":[1,2,3,5,6],

              "max_features":['auto','sqrt','log2',None],

              

              "criterion": ["gini", "entropy"],

              

             }



n_iter_search = 20

random_search = RandomizedSearchCV(clfETC, param_distributions=param_dist,

                                   n_iter=n_iter_search, cv=10,verbose=1)



#Already Searched Parameter

random_search.fit(X_frequencyVec, df.label)

print(random_search.best_params_)
best_params=random_search.best_params_

print(best_params)

min_samples_leaf=best_params['min_samples_leaf']

max_features=best_params['max_features']

max_depth=best_params['max_depth']

criterion=best_params['criterion']
X_train, X_test, y_train, y_test = train_test_split(X_frequencyVec, df.label, test_size=.3)

from sklearn.ensemble import RandomForestClassifier

clfET = tree.ExtraTreeClassifier( min_samples_split= min_samples_leaf, max_features=max_features,max_depth= max_depth,splitter= 'best',

                              criterion= criterion)

                            



from sklearn.metrics import confusion_matrix

clfET=clfET.fit(X_train,y_train)

X_test=np.asarray(X_test)

y_pred=clfET.predict(X_test)

true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_test,y_pred).ravel()



print("true_negative: ",true_negative)

print("false_positive: ",false_positive)

print("false_negative: ",false_negative)

print("true_positive: ",true_positive)

print("\n\n Accuracy Measures\n\n")

Accuracy=(true_positive+true_negative)/(true_positive+false_negative+true_negative+false_positive)

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

from sklearn.model_selection import RandomizedSearchCV

from sklearn import tree





clfETC=RandomForestClassifier()





param_dist = {"max_depth": [30,50,23,25,17,None],

               "max_features": [0.03,0.07,0.10,0.15,0.20,0.30,"auto"],

              "min_samples_split": [2,3,4,5,6,7,8,9],

              "n_estimators":[200,300,400,500,600,700],

              "criterion": ["gini", "entropy"],

              "bootstrap":[True,False]

              

             

             }



n_iter_search = 20

random_search = RandomizedSearchCV(clfETC, param_distributions=param_dist,

                                   n_iter=n_iter_search, cv=10,verbose=1)



#Already Searched Parameter

random_search.fit(X_frequencyVec, df.label)

print(random_search.best_params_)
best_params=random_search.best_params_

print(best_params)

n_estimators=best_params['n_estimators']

min_samples_split=best_params['min_samples_split']

max_features=best_params['max_features']

max_depth=best_params['max_depth']

criterion=best_params['criterion']

bootstrap=best_params['bootstrap']

X_train, X_test, y_train, y_test = train_test_split(X_frequencyVec, df.label, test_size=.3, random_state=0)

from sklearn.ensemble import RandomForestClassifier

clfRF = RandomForestClassifier(n_estimators= n_estimators, min_samples_split= min_samples_split, max_features= max_features,

                             max_depth= max_depth, criterion= criterion, bootstrap=bootstrap

                             )

                            



from sklearn.metrics import confusion_matrix

clfRF=clfRF.fit(X_train,y_train)

X_test=np.asarray(X_test)

y_pred=clfRF.predict(X_test)

true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_test,y_pred).ravel()



print("true_negative: ",true_negative)

print("false_positive: ",false_positive)

print("false_negative: ",false_negative)

print("true_positive: ",true_positive)

print("\n\n Accuracy Measures\n\n")

Accuracy=(true_positive+true_negative)/(true_positive+false_negative+true_negative+false_positive)

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



from sklearn.model_selection import RandomizedSearchCV

from sklearn import tree





clfBC=ensemble.BaggingClassifier()





param_dist = {"base_estimator": [clfRF,clfET],

               "n_estimators": [30,60,100,150,200,250],

              "warm_start": [True,False] ,

              "bootstrap":[True,False]

              

             

             }



n_iter_search = 20

random_search = RandomizedSearchCV(clfBC, param_distributions=param_dist,

                                   n_iter=n_iter_search, cv=10,verbose=1)



#Already Searched Parameter

random_search.fit(X_frequencyVec, df.label)

print(random_search.best_params_)
best_params=random_search.best_params_

print(best_params)

base_estimator=best_params['base_estimator']

n_estimators=best_params['n_estimators']

warm_start=best_params['warm_start']

bootstrap=best_params['bootstrap']

#best params

#{'warm_start': True, 'n_estimators': 30, 'bootstrap': True, 'base_estimator': None}
bstlfy=ensemble.BaggingClassifier(base_estimator=base_estimator,n_estimators=n_estimators,warm_start=warm_start,bootstrap=bootstrap)

bstlfy=bstlfy.fit(X_train, y_train)

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
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split Data

lb = LabelBinarizer()

Y = lb.fit_transform(df.label)



X_train, X_test,y_train,y_test = train_test_split(df.sequences, Y, test_size = 0.2, random_state =0)













y_test_cat=keras.utils.to_categorical(y_test)

y_train_cat=keras.utils.to_categorical(y_train)

# Create a Count Vectorizer to gather the unique elements in sequence

vect = CountVectorizer(analyzer = 'char_wb', ngram_range = (4,4))



# Fit and Transform CountVectorizer

vect.fit(X_train)

X_train_df = vect.transform(X_train)

X_test_df = vect.transform(X_test)



#Print a few of the features

print(vect.get_feature_names()[-20:])

from sklearn.model_selection import RandomizedSearchCV

from sklearn import tree





clfETC=linear_model.LogisticRegressionCV()



param_dist = {"Cs": [10,5,20,15,30,4,3],

               "fit_intercept": [True,False],

              "tol":[0.0001,0.000001,0.01,0.001],

              

              "solver": ["newton-cg", "lbfgs","liblinear","saga"],

              

              "n_jobs":[-1],

              "multi_class":["ovr","auto"]

             

             }



n_iter_search = 20

random_search = RandomizedSearchCV(clfETC, param_distributions=param_dist,

                                   n_iter=n_iter_search, cv=10,verbose=1)



#Already Searched Parameter

random_search.fit(X_frequencyVec, df.label)

print(random_search.best_params_)

#previous best params

#tol=1e-06, solver= 'liblinear', n_jobs= -1, multi_class= 'ovr', fit_intercept= True, Cs=5

#88% accuracy




clfRF = linear_model.LogisticRegressionCV(tol=1e-06, solver= 'liblinear', n_jobs= -1, multi_class= 'ovr', fit_intercept= True, Cs=5)

                            

from sklearn.metrics import confusion_matrix

clfRF=clfRF.fit(X_train_df,y_train)

X_test=np.asarray(X_test_df)

y_pred=clfRF.predict(X_test_df)

true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_test,y_pred).ravel()







print("true_negative: ",true_negative)

print("false_positive: ",false_positive)

print("false_negative: ",false_negative)

print("true_positive: ",true_positive)

print("\n\n Accuracy Measures\n\n")

Accuracy=(true_positive+true_negative)/(true_positive+false_negative+true_negative+false_positive)

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

bstlfy=ensemble.BaggingClassifier(base_estimator=clfRF,n_estimators=10)

bstlfy=bstlfy.fit(X_train_df, y_train)

prediction=bstlfy.predict(X_test_df)

    



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
Ada=ensemble.AdaBoostClassifier(base_estimator=bstlfy, n_estimators=40, learning_rate=1.0, algorithm="SAMME", random_state=None)

Ada=Ada.fit(X_train_df, y_train)

prediction=Ada.predict(X_test_df)

    



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
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split Data

lb = LabelBinarizer()

y = lb.fit_transform(df.label)



X=df.sequences















vect = CountVectorizer(analyzer = 'char_wb', ngram_range = (4,4))



# Fit and Transform CountVectorizer

vect.fit(X)

X = vect.transform(X)





#Print a few of the features

print(vect.get_feature_names()[-20:])





kfold = StratifiedKFold(n_splits=100, shuffle=True)

cvscores = []



iterator = 1

cv_score = 0

for train, test in kfold.split(X, y):

    print('Fold : '+str(iterator))

    # giving 100% accuracy on n_estimator 50

    RandomForest = linear_model.LogisticRegressionCV(tol=1e-06, solver= 'liblinear', n_jobs= -1, multi_class= 'ovr', fit_intercept= True, Cs=5).fit(X[train],y[train].ravel())

    pred = np.round(RandomForest.predict(X[test]))

    tn, fp, fn, tp = confusion_matrix(y[test], pred, labels=[1,0]).ravel()

    acc = np.round(((tn+tp)/(tn+fp+fn+tp))*100, 2)

    cvscores.append([tn,fp,fn,tp,acc])

    iterator=iterator+1

    print([tn,fp,fn,tp,acc])

    cv_score = cv_score + acc



print('\n\rFinal JackKnife Score = ', np.round(cv_score/kfold.n_splits,2))
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split Data

lb = LabelBinarizer()

y = lb.fit_transform(df.label)



X=df.sequences















vect = CountVectorizer(analyzer = 'char_wb', ngram_range = (4,4))



# Fit and Transform CountVectorizer

vect.fit(X)

X = vect.transform(X)





#Print a few of the features

print(vect.get_feature_names()[-20:])





kfold = StratifiedKFold(n_splits=100, shuffle=True)

cvscores = []



iterator = 1

cv_score = 0

for train, test in kfold.split(X, y):

    print('Fold : '+str(iterator))

    # giving 100% accuracy on n_estimator 50

    RandomForest = ensemble.BaggingClassifier(base_estimator=clfRF,n_estimators=10).fit(X[train],y[train].ravel())

    pred = np.round(RandomForest.predict(X[test]))

    tn, fp, fn, tp = confusion_matrix(y[test], pred, labels=[1,0]).ravel()

    acc = np.round(((tn+tp)/(tn+fp+fn+tp))*100, 2)

    cvscores.append([tn,fp,fn,tp,acc])

    iterator=iterator+1

    print([tn,fp,fn,tp,acc])

    cv_score = cv_score + acc



print('\n\rFinal JackKnife Score = ', np.round(cv_score/kfold.n_splits,2))
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split Data

lb = LabelBinarizer()

y = lb.fit_transform(df.label)



X=df.sequences















vect = CountVectorizer(analyzer = 'char_wb', ngram_range = (4,4))



# Fit and Transform CountVectorizer

vect.fit(X)

X = vect.transform(X)





#Print a few of the features

print(vect.get_feature_names()[-20:])





kfold = StratifiedKFold(n_splits=100, shuffle=True)

cvscores = []



iterator = 1

cv_score = 0

for train, test in kfold.split(X, y):

    print('Fold : '+str(iterator))

    # giving 100% accuracy on n_estimator 50

    RandomForest = ensemble.AdaBoostClassifier(base_estimator=bstlfy, n_estimators=40, learning_rate=1.0, algorithm="SAMME", random_state=None).fit(X[train],y[train].ravel())

    pred = np.round(RandomForest.predict(X[test]))

    tn, fp, fn, tp = confusion_matrix(y[test], pred, labels=[1,0]).ravel()

    acc = np.round(((tn+tp)/(tn+fp+fn+tp))*100, 2)

    cvscores.append([tn,fp,fn,tp,acc])

    iterator=iterator+1

    print([tn,fp,fn,tp,acc])

    cv_score = cv_score + acc



print('\n\rFinal JackKnife Score = ', np.round(cv_score/kfold.n_splits,2))