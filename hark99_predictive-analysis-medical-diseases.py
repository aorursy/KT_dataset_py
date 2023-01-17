# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import preprocessing

from sklearn.model_selection import train_test_split





# Maximazing the Dsiplay

pd.set_option('display.max_columns', None) 

pd.set_option('display.max_rows', None)

pd.set_option('display.width', None)





import warnings

import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)



from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Defining the Columns

col_names=['age','gender','Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']
df=pd.read_excel('/kaggle/input/Medical_diseases_data.xlsx',names=col_names)

df.head()
df.shape
# Exploring the type of each column for numeric conversaion if required

df.dtypes
# Converting object to numeric value for desgining algorithms

from sklearn.preprocessing import LabelEncoder

for f in ['gender']:

    lbl = LabelEncoder()

    lbl.fit(list(df[f].values) + list(df[f].values))

    df[f] = lbl.transform(list(df[f].values))

# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)

X[0:5]
# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])

y [0:5]
!pip install iterative-stratification
#!pip install iterative-stratification

print ('CLASSIFIER CHAINS')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training an test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')





# Designing 



# using classifier chains

from skmultilearn.problem_transform import ClassifierChain

from sklearn.naive_bayes import GaussianNB



# initialize classifier chains multi-label classifier

# with a gaussian naive bayes base classifier

classifier = ClassifierChain(GaussianNB())



# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)



print ("PREDICTIVE MODELLING OUTCOME")

print (predictions)

print ('=====================================')



# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions),'The above is classification_report for Stratification method')

from sklearn.metrics import f1_score,recall_score, precision_score

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('=====================================')

print ('Accuracy (Stratification):', accuracy_score(y_test,predictions))

print ('Hamming Loss (Stratification):' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')





# ITERATIVE METHOD

print ('ITERATIVE METHOD FOR MULTI-CLASS SPLITTING')

print ('=====================================')

# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training an test set

from skmultilearn.model_selection import iterative_train_test_split

X_train, y_train, X_test, y_test = iterative_train_test_split(X, y,test_size = 0.2)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')



# Designing 



# using classifier chains

from skmultilearn.problem_transform import ClassifierChain

from sklearn.naive_bayes import GaussianNB



# initialize classifier chains multi-label classifier

# with a gaussian naive bayes base classifier

classifier = ClassifierChain(GaussianNB())



# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)

print ("PREDICTIVE MODELLING OUTCOME")

print (predictions)

print ('=====================================')



# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions),'The above is classification report for iterative')

print ('=====================================')

from sklearn.metrics import f1_score,recall_score, precision_score

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('=====================================')

print ('Accuracy (Iterative):', accuracy_score(y_test,predictions))

print ('Hamming Loss (Iterative):' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')

print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')
print ('SVM (Classifier Chains)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')





from skmultilearn.problem_transform import ClassifierChain

from sklearn.svm import SVC



# initialize Classifier Chain multi-label classifier

# with an SVM classifier

# SVM in scikit only supports the X matrix in sparse representation







classifier = ClassifierChain(

    classifier = SVC(),

    require_dense = [False, True]

)





# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)



# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, f1_score , precision_score,recall_score  

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')



print ('RANDOM FOREST (Classifier Chains)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')





from skmultilearn.problem_transform import ClassifierChain

from sklearn.ensemble import RandomForestClassifier



# initialize ClassifierChain multi-label classifier with a RandomForest

classifier = ClassifierChain(

    classifier = RandomForestClassifier(n_estimators=100),

    require_dense = [False, True]

)



# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)





# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')

print ('DECISION TREE CLASSIFIER, C4.5 (Classifier Chains)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')



from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)



# predict

predictions = clf.predict(X_test)





# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')
# using Label Powerset

from skmultilearn.problem_transform import LabelPowerset

print ('SVM ( LabelPowerset)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')





from skmultilearn.problem_transform import LabelPowerset

from sklearn.svm import SVC



# initialize Classifier Chain multi-label classifier

# with an SVM classifier

# SVM in scikit only supports the X matrix in sparse representation







classifier = LabelPowerset(

    classifier = SVC(),

    require_dense = [False, True]

)





# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)



# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')



print ('RANDOM FOREST ( LabelPowerset)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')





from skmultilearn.problem_transform import LabelPowerset

from sklearn.ensemble import RandomForestClassifier



# initialize ClassifierChain multi-label classifier with a RandomForest

classifier = LabelPowerset(

    classifier = RandomForestClassifier(n_estimators=100),

    require_dense = [False, True]

)



# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)





# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')

print ('DECISION TREE CLASSIFIER, C4.5 ( LabelPowerset)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')



from sklearn import tree

classifier = LabelPowerset(

    classifier = tree.DecisionTreeClassifier(),

    require_dense = [False, True]

)





# train

classifier = classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)





# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')
from skmultilearn.ensemble import RakelD



print ('SVM ( RakelD)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')





from sklearn.svm import SVC



# initialize Classifier Chain multi-label classifier

# with an SVM classifier

# SVM in scikit only supports the X matrix in sparse representation









classifier = RakelD(

    base_classifier=SVC(),

    base_classifier_require_dense=[False, True],

    labelset_size=4

)



# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)



# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')



print ('RANDOM FOREST (RakelD)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')







from sklearn.ensemble import RandomForestClassifier



# initialize ClassifierChain multi-label classifier with a RandomForest

classifier = RakelD(

    base_classifier=RandomForestClassifier(n_estimators=100),

    base_classifier_require_dense=[False, True],

    labelset_size=4

)







# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)





# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')

print ('DECISION TREE CLASSIFIER, C4.5(RakelD)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')



from sklearn import tree

classifier = RakelD(

    base_classifier=tree.DecisionTreeClassifier(),

    base_classifier_require_dense=[False, True],

    labelset_size=4

)





# train

classifier = classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)





# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')
from sklearn.naive_bayes import MultinomialNB

classifier = ClassifierChain(

    classifier = MultinomialNB(),

    require_dense = [False, True]

)





# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)



# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')
print ('MLkNN')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training an test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')



# MLkNN 



from skmultilearn.adapt import MLkNN



classifier = MLkNN(k=10,s=1)



# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)



print ('=====================================')



# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')





print ('BINARY RELEVANCE kNN')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')



from skmultilearn.adapt import BRkNNaClassifier



classifier = BRkNNaClassifier(k=10)



# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)





# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder

graph_builder = LabelCooccurrenceGraphBuilder(weighted=True,

                                              include_self_edges=False)



label_names=[i for i in range(3)]

edge_map = graph_builder.transform(y_train)

print("{} labels, {} edges".format(len(label_names), len(edge_map)))
from skmultilearn.cluster import NetworkXLabelGraphClusterer



# we define a helper function for visualization purposes

def to_membership_vector(partition):

    return {

        member :  partition_id

        for partition_id, members in enumerate(partition)

        for member in members

    }

clusterer = NetworkXLabelGraphClusterer(graph_builder, method='louvain')
partition = clusterer.fit_predict(X_train,y_train)

membership_vector = to_membership_vector(partition)

print('There are', len(partition),'clusters')
import networkx as nx

names_dict = dict(enumerate(x for x in label_names))

import matplotlib.pyplot as plt

%matplotlib inline

nx.draw(

    clusterer.graph_,

    pos=nx.spring_layout(clusterer.graph_,k=4),

    labels=names_dict,

    with_labels = True,

    width = [10*x/y_train.shape[0] for x in clusterer.weights_['weight']],

    node_color = [membership_vector[i] for i in range(y_train.shape[1])],

    cmap=plt.cm.viridis,

    node_size=250,

    font_size=10,

    font_color='white',

    alpha=0.8

)
from skmultilearn.ensemble import LabelSpacePartitioningClassifier

from skmultilearn.cluster import LabelCooccurrenceGraphBuilder

from skmultilearn.cluster import NetworkXLabelGraphClusterer



print ('SVM ( ECC)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')





from sklearn.svm import SVC



# initialize Classifier Chain multi-label classifier

# with an SVM classifier

# SVM in scikit only supports the X matrix in sparse representation



# Label Graph

# When the label space is large, we can try to explore it using graph methods.

# Each label is a node in the graph and an edge exists when labels co-occur, 

# weighted by the frequency of co-occurrence.



graph_builder = LabelCooccurrenceGraphBuilder(weighted=True,

                                              include_self_edges=False)



edge_map = graph_builder.transform(y_train)



classifier = LabelSpacePartitioningClassifier(

    classifier = ClassifierChain(

    classifier = SVC(),

    require_dense = [False, True]

),

    clusterer  = NetworkXLabelGraphClusterer(graph_builder, method='louvain')

)



# train

classifier.fit(X_train,y_train)



# predict

predictions=classifier.predict(X_test)



# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')



print ('RANDOM FOREST (ECC)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')







graph_builder = LabelCooccurrenceGraphBuilder(weighted=True,

                                              include_self_edges=False)



edge_map = graph_builder.transform(y_train)



classifier = LabelSpacePartitioningClassifier(

    classifier = ClassifierChain(

    classifier = RandomForestClassifier(n_estimators=100),

    require_dense = [False, True]

),

    clusterer  = NetworkXLabelGraphClusterer(graph_builder, method='louvain')

)



# train

classifier.fit(X_train,y_train)



# predict

predictions=classifier.predict(X_test)



# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')

print ('DECISION TREE CLASSIFIER, C4.5(ECC)')

print ('=====================================')

# STRATIFICATION METHOD

print ("STRATIFICATION CROSS VALIDATION")

print ('=====================================')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=10, random_state=0)



# Selecting Features 

feature_df = df[['age','gender',\

           'Systolic (mmHg)','Diastolic (mmHg)','Mean Arterial Pressure (MAP) (mmHg)',\

           'Fasting Blood Glucose (mg/dL)','Random (mg/dL)', 'HbA1C (%)',\

           'eAG (Estimate Average Glucose) Level (mg/dL)','Total cholesterol (mg/dL)',' HDL (mg/dL)',\

           'LDL (mg/dL)','Triglyceride  (mg/dL)','VLDL  (mg/dL)','Serum Creatinine  (mg/dL)','BUN  (mg/dL)']]

X = np.asarray(feature_df)



# target variable

y = np.asarray(df[['Diabetes_Mllitus','Hypertension Diagnosis','Hyperlipidemia Diagnosis']])



# Splitting data into training and test set

for train_index, test_index in mskf.split(X, y):

   #print("TRAIN:", train_index, "TEST:", test_index)

   X_train, X_test = X[train_index], X[test_index]

   y_train, y_test = y[train_index], y[test_index]

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

print ('=====================================')





graph_builder = LabelCooccurrenceGraphBuilder(weighted=True,

                                              include_self_edges=False)



edge_map = graph_builder.transform(y_train)



classifier = LabelSpacePartitioningClassifier(

    classifier = ClassifierChain(

    classifier = tree.DecisionTreeClassifier(),

    require_dense = [False, True]

),

    clusterer  = NetworkXLabelGraphClusterer(graph_builder, method='louvain')

)



# train

classifier.fit(X_train,y_train)



# predict

predictions=classifier.predict(X_test)





# Evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics

print ('EVALUATION')

print (classification_report(y_test,predictions))

print ('=====================================')

print ('f1_score (micro)',f1_score(y_test, predictions, average='micro') )

print ('recall_score (micro)',recall_score(y_test, predictions, average='micro') )

print ('precision_score (micro)',precision_score(y_test, predictions, average='micro') )

print ('f1_score (macro)',f1_score(y_test, predictions, average='macro') )

print ('recall_score (macro)',recall_score(y_test, predictions, average='macro') )

print ('precision_score (macro)',precision_score(y_test, predictions, average='macro') )

print ('Accuracy:', accuracy_score(y_test,predictions))

print ('Hamming Loss:' , metrics.hamming_loss(y_test, predictions))

print ('=====================================')



print ('Note: In Multitask, the accuracy is always be in the form of sub-accuracy.')

print ('We used the same way as we do for finding the accuracy through sklearn metric.')

print ('=====================================')