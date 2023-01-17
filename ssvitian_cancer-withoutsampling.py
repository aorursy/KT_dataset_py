import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import re

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss, accuracy_score

from sklearn.svm import SVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer

from keras.utils.np_utils import to_categorical

from sklearn.ensemble import VotingClassifier
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
training_text = pd.read_csv("../input/cancer-diagnosis/training_text",sep="\|\|", header=None, skiprows=1, names=["ID","Text"])
training_text.head(5)
training_text.shape
training_variants = pd.read_csv("../input/cancer-diagnosis/training_variants")
training_variants.head(5)
training_variants.shape
#Merging variants and text on ID
training_merge = training_variants.merge(training_text,left_on="ID",right_on="ID")
training_merge.head(5)

#Now we have the data
training_merge.shape
test_text = pd.read_csv("../input/cancertreatment/stage2_test_text.csv",sep="\|\|", header=None, skiprows=1, names=["ID","Text"])
test_text.shape
test_variants = pd.read_csv("../input/cancertreatment/stage2_test_variants.csv")
test_variants.shape
test_merge = test_variants.merge(test_text,left_on="ID",right_on="ID")
# Let's understand the type of values present in each column of our dataframe 'train_merge' dataframe.

test_merge.info()
training_merge.describe(include='all')
training_merge.isnull().sum()
training_merge.columns
training_merge["Text_num_words"] = training_merge["Text"].apply(lambda x: len(str(x).split()) )

training_merge["Text_num_chars"] = training_merge["Text"].apply(lambda x: len(str(x)) )
training_merge['Variation'].describe()
training_merge['Gene'].describe()
from collections import Counter

# Import WordNetLemmatizer

from nltk.stem import WordNetLemmatizer

# Import word_tokenize

from nltk.tokenize import word_tokenize

# Import stopwords

from nltk.corpus import stopwords

# Import string

import string

#Importing 
#Tokenzing-splitting up a larger body of text into smaller lines, words or even creating words 
#imputing gene row value to null data of text rows as for all other columns, Gene values are present in Text data

training_merge['Text'] = training_merge.apply(lambda row: row['Gene'] if pd.isnull(row['Text']) else row['Text'],axis=1)
training_merge.isnull().sum()
#imputing gene row value to null data of text rows as for all other columns, Gene values are present in Text data

test_merge['Text'] = test_merge.apply(lambda row: row['Gene'] if pd.isnull(row['Text']) else row['Text'],axis=1)
mincl=[3,5,6,8,9]

maxcl=[1,2,4,7]
dfA=training_merge[training_merge['Class'].isin(mincl)]

dfB=training_merge[training_merge['Class'].isin(maxcl)]
dfA.head(5)
dfA.describe()
dfB.describe()
#taking class column as dependent variable ie which needs to be find out from all other columns in our data

ym=training_merge.Class

yA=dfA.Class

yB=dfB.Class
X_A=dfA[["Text","Variation","Gene"]]

X_B=dfB[["Text","Variation","Gene"]]

X_m=training_merge[["Text","Variation","Gene"]]
X_A.head()
X_B.head()
# Definig vectorizing object for Text column

vect_text= CountVectorizer(stop_words ='english')



#Defining vectorizing object for Variation column

vect_variation= CountVectorizer(stop_words ='english')



##Defining vectorizing object for Gene column

gene_variation= CountVectorizer(stop_words ='english')
#vectorizing  for Text column which gives the count of repeated words for each row for both the dataframes

vect_text.fit(X_A["Text"])

vect_text.fit(X_B["Text"])

vect_text.fit(X_m["Text"])
#vectorizing for Variation column  which gives the count of repeated words for each row

vect_variation.fit(X_A["Variation"])

vect_variation.fit(X_B["Variation"])

vect_variation.fit(X_m["Variation"])
gene_variation.fit(X_A["Gene"])

gene_variation.fit(X_B["Gene"])

gene_variation.fit(X_m["Gene"])
len(vect_text.vocabulary_)
len(vect_variation.vocabulary_)
len(gene_variation.vocabulary_)
vect_text.vocabulary_
vect_variation.vocabulary_
gene_variation.vocabulary_
#transforming count of Variation words in to matrix

variation_tranform_train_A=vect_variation.transform(X_A["Variation"])

variation_tranform_train_B=vect_variation.transform(X_B["Variation"])

variation_tranform_train_m=vect_variation.transform(X_m["Variation"])
#transforming count of Text words in to matrix

text_transformed_train_A= vect_text.transform(X_A["Text"])

text_transformed_train_B= vect_text.transform(X_B["Text"])

text_transformed_train_m= vect_text.transform(X_m["Text"])
#transforming count of gene words in to matrix

gene_transformed_train_A= gene_variation.transform(X_A["Gene"])

gene_transformed_train_B= gene_variation.transform(X_B["Gene"])

gene_transformed_train_m= gene_variation.transform(X_m["Gene"])
#merging train data of two Matrix horixzontally to train the model

import scipy.sparse as sp

XA_final = sp.hstack((variation_tranform_train_A,text_transformed_train_A,gene_transformed_train_A))

XB_final = sp.hstack((variation_tranform_train_B,text_transformed_train_B,gene_transformed_train_B))

Xm_final = sp.hstack((variation_tranform_train_m,text_transformed_train_m,gene_transformed_train_m))
XA_final.shape
XB_final.shape
yA.shape
yB.shape
Xm_final.shape
ym.shape
# splitting into test and train

from sklearn.model_selection  import train_test_split

from imblearn.over_sampling import SMOTE

XA_train, XA_test, yA_train, yA_test = train_test_split(XA_final, yA, random_state=1)
XB_train, XB_test, yB_train, yB_test = train_test_split(XB_final, yB, random_state=1)
print(XA_train.shape)
print(yA_train.shape)
print(XA_test.shape)
print(yA_test.shape)
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report 

# Model Generation Using Multinomial Naive Bayes

clfA = MultinomialNB().fit(XA_train, yA_train)

predicted= clfA.predict(XA_test)

print("MultinomialNB Accuracy:",metrics.accuracy_score(yA_test, predicted))

print(classification_report(yA_test,predicted))
# Model Generation Using Multinomial Naive Bayes

clfB = MultinomialNB().fit(XB_train, yB_train)

predicted= clfB.predict(XB_test)

print("MultinomialNB Accuracy:",metrics.accuracy_score(yB_test, predicted))

print(classification_report(yB_test,predicted))
from sklearn.ensemble import VotingClassifier
#Using Average weighting on the models that we have generated

#Using a voting classifier on Multinomial NB after hyperparameter tuning would be a waste as it NB is a pretty simple Model
Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm_final, ym, random_state=1)
# Model Generation Using Multinomial Naive Bayes

clfm = MultinomialNB().fit(Xm_train, ym_train)

predict= clfm.predict(Xm_test)

print("MultinomialNB Accuracy:",metrics.accuracy_score(ym_test, predict))

print(classification_report(ym_test,predict))
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn import datasets

from sklearn.multiclass import OneVsRestClassifier
#We see that the accuracy is skewed for the entire model as the data is imbalanced as we saw in the class distribution
#We would now try to tune the SVM hyperparameters and stack the classifiers built from them to be able to aid out 

#our final machine learning model
#Tuning the hyperparameters for the minority data
from sklearn.model_selection import GridSearchCV 

  

# defining parameter range 

param_grid = {'C': [0.1, 1], #10, 100, 1000],  

              'gamma': [1, 0.1, 0.01], #0.001, 0.0001], 

              'kernel': ['linear']}  

  

gridA = GridSearchCV(SVC(probability=True), param_grid, refit = True, verbose = 3,) 

  

# fitting the model for grid search 

gridA.fit(XA_train, yA_train) 
# print best parameter after tuning 

print(gridA.best_params_) 

  

# print how our model looks after hyper-parameter tuning 

print(gridA.best_estimator_) 
gridA_predictions = gridA.predict(XA_test) 

  

# print classification report 

print(classification_report(yA_test, gridA_predictions)) 
# defining parameter range 

param_grid = {'C': [0.1, 1], #10, 100, 1000],  

              'gamma': [1],

              'kernel': ['linear']}  

gridB = GridSearchCV(SVC(probability=True), param_grid, refit = True, verbose = 3) 

  

# fitting the model for grid search 

gridB.fit(XB_train, yB_train) 
gridB_predictions = gridB.predict(XB_test) 

  

# print classification report 

print(classification_report(yB_test, gridB_predictions)) 
eclf2 = VotingClassifier(estimators=[('svmA',gridA), ('svmB', gridB)], voting='soft')
# fitting the model for grid search 

eclf2.fit(Xm_train, ym_train) 
#eclf2.probability = True
ym_pred=eclf2.predict(Xm_test)
print(classification_report(ym_test, ym_pred))
for i in range(1,9):

    print("Before SMOTE, counts of label {}: {}".format(i,sum(ym_train == i))) 

 

# apply near miss 

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2) 

  

X_train_miss, y_train_miss = sm.fit_sample(Xm_train, ym_train.ravel()) 

  

print('After SMOTE, the shape of train_X:{}'.format(X_train_miss.shape)) 

print('After SMOTE, the shape of train_y: {} \n'.format(y_train_miss.shape)) 



for i in range(1,9):

    print("After SMOTE, counts of label {}: {}".format(i,sum(y_train_miss == i))) 
#Hyperparameter tuning for Random Forest
ran_clf=RandomForestClassifier(bootstrap=True,

                                                             class_weight=None,

                                                             criterion='gini',

                                                             max_depth=10,

                                                             max_features='auto',

                                                             max_leaf_nodes=None,

                                                             min_impurity_decrease=0.0,

                                                             min_impurity_split=None,

                                                             min_samples_leaf=1,

                                                             min_samples_split=2,

                                                             min_weight_fraction_leaf=0.0,

                                                             n_estimators=200,

                                                             n_jobs=None,

                                                             oob_score=False,

                                                             random_state=42,

                                                             verbose=0,

                                                             warm_start=False)
ran_clf.fit(Xm_train,ym_train_miss)

y_pred=ran_clf.predict(Xm_test)

print(classification_report(ym_test,ym_pred))
voting_clf = VotingClassifier([('svc', svm.SVC(probability=True)),

                            ('nsb', MultinomialNB()),

                            ('rfor', RandomForestClassifier())],voting='soft')
voting_clf.fit(Xm_train_miss, ym_train_miss)
#y_pred_class=voting_clf.predict_proba(Xm_test)

classification_report(ym_test, y_pred_class)
y_pred_class