# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, warnings, gc

warnings.filterwarnings("ignore")



# SKLearn (feature selection)

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import SelectFpr, SelectFdr, SelectFwe

from sklearn.feature_selection import GenericUnivariateSelect

from sklearn.feature_selection import SelectFromModel



# SKLearn (feature extraction)

from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer







# SkLearn Classification Algorithm

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



#Sklearn Metrics, Model Selection , Preprocessing & Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.pipeline import Pipeline



# Tabulate

from tabulate import tabulate



# PLot

import matplotlib.pyplot as plt

import seaborn as sns

# Load Data

url = '../input/all-datasets-for-practicing-ml/Class/Class_Abalone.csv'

data = pd.read_csv(url, header='infer')
# Label Encoding

encoder = LabelEncoder()

data['Sex']= encoder.fit_transform(data['Sex']) 



# Seperate Features & Target

columns = data.columns

target = ['Sex']

features = columns[1:]



X = data[features]

y = data[target]
# Inspect

X.head()
'''Classification Model Evaluation Using the Above Data'''

'''Classifier Model = Random Forest '''



# Dataset Split

''' Training = 90% & Validation = 10%  '''

test_size = 0.1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True) 





'''Creating Pipeline [Feature Scaling & Classification]'''

pipe = Pipeline([

         ("Feature_Scaling", StandardScaler()),

         ("RandomForest_Classifier", RandomForestClassifier(random_state=1, verbose=0)) ])



pipe.fit(X_train, y_train)





# Accuracy

print("Random Forest Accuracy (without feature selection): ", '{:.2%}'.format(pipe.score(X_val,y_val)))
'''Applying SelectKBest with Scoring Function = Chi2 to Abalone dataset to retrieve the **4** best features.'''



X_new = pd.DataFrame(SelectKBest(chi2, k=4).fit_transform(X,y))



#Inspect

X_new.head()
#Rename

X_new.rename(columns={0: "Whole_Weight", 1: "Shucked_Weight", 

                      2:"Shell_Weight", 3: "Rings"}, inplace=True)



#Inspect

X_new.head()
'''Classification Model Evaluation Using the Newly Selected Feature Data'''



# Dataset Split

''' Training = 90% & Validation = 10%  '''

test_size = 0.1

X_train, X_val, y_train, y_val = train_test_split(X_new, y, test_size=test_size, random_state=0, shuffle=True) 



# Feature Scale & Train Model

pipe.fit(X_train, y_train)





# Tabulate Data Empty List

tab_data = []



# Accuracy

#print("Random Forest Accuracy (with feature selection): ", '{:.2%}'.format(pipe.score(X_val,y_val)))

tab_data.append(['Chi2', '{:.2%}'.format(pipe.score(X_val,y_val))])

'''Applying SelectKBest with Scoring Function = f_classif to Abalone dataset to retrieve the **4** best features.'''



X_new = pd.DataFrame(SelectKBest(f_classif, k=4).fit_transform(X,y))



# Dataset Split

''' Training = 90% & Validation = 10%  '''

test_size = 0.1

X_train, X_val, y_train, y_val = train_test_split(X_new, y, test_size=test_size, random_state=0, shuffle=True) 



# Feature Scale & Train Model

pipe.fit(X_train, y_train)



# Accuracy

#print("Random Forest Accuracy (with feature selection): ", '{:.2%}'.format(pipe.score(X_val,y_val)))

tab_data.append(['f_classif', '{:.2%}'.format(pipe.score(X_val,y_val))])
'''Applying SelectKBest with Scoring Function = mutual_info_classif to Abalone dataset to retrieve the **4** best features.'''



X_new = pd.DataFrame(SelectKBest(mutual_info_classif, k=4).fit_transform(X,y))



# Dataset Split

''' Training = 90% & Validation = 10%  '''

test_size = 0.1

X_train, X_val, y_train, y_val = train_test_split(X_new, y, test_size=test_size, random_state=0, shuffle=True) 



# Feature Scale & Train Model

pipe.fit(X_train, y_train)



# Accuracy

#print("Random Forest Accuracy (with feature selection): ", '{:.2%}'.format(pipe.score(X_val,y_val)))

tab_data.append(['mutual_info_classif', '{:.2%}'.format(pipe.score(X_val,y_val))])
# Tabulate Data

print("Random Forest Accuracy (with SelectKBest feature selection):\n\n", tabulate(tab_data, headers=['Scoring_Func', 'Accuracy']))
'''Applying SelectPercentile with Scoring Function = Chi2 to Abalone dataset to retrieve the **4** best features.'''

''' Percentile = Percent of features to keep'''

'''For this example, we're going to keep 70% of the features'''



X_new = pd.DataFrame(SelectPercentile(chi2, percentile=70).fit_transform(X,y))



#Inspect

X_new.shape
'''Classification Model Evaluation Using the Newly Selected Feature Data'''



# Dataset Split

''' Training = 90% & Validation = 10%  '''

test_size = 0.1

X_train, X_val, y_train, y_val = train_test_split(X_new, y, test_size=test_size, random_state=0, shuffle=True) 



# Feature Scale & Train Model

pipe.fit(X_train, y_train)



# Accuracy

print("Random Forest Accuracy (with SelectPercentile{Chi2} feature selection): ", '{:.2%}'.format(pipe.score(X_val,y_val)))
'''Applying FPR with Scoring Function = Chi2 to Abalone dataset to retrieve the best features.'''



X_new = pd.DataFrame(SelectFpr(chi2).fit_transform(X,y))



#Inspect

X_new.shape
'''Applying FDR with Scoring Function = Chi2 to Abalone dataset to retrieve the best features.'''



X_new = pd.DataFrame(SelectFdr(chi2).fit_transform(X,y))



#Inspect

X_new.shape
'''Applying FWE with Scoring Function = Chi2 to Abalone dataset to retrieve the best features.'''



X_new = pd.DataFrame(SelectFwe(chi2).fit_transform(X,y))



#Inspect

X_new.shape
'''Applying GenericUnivariateSelect with Scoring Function = Chi2 & Mode = k_best to Abalone dataset to retrieve the best features.'''



X_new = pd.DataFrame(GenericUnivariateSelect(chi2, mode='k_best', param=4).fit_transform(X,y))



#Inspect

X_new.shape
# Dict

staff = [{'name': 'John Oxboro', 'age': 23, 'role':'Manager'},

         {'name': 'Regina Smith', 'age': 10, 'role':'Lead'},

         {'name': 'Ollie Dyson', 'age': 28, 'role':'Architect'},

        {'name': 'Ian McGrath', 'age': 48, 'role':'Engineer'}]



# Convert Dictionary To Feature Matrix

dv = DictVectorizer()

dv.fit_transform(staff).toarray()



#View Feature Names

dv.get_feature_names()
# List of Texts

text = ['The quick brown fox jumped over the lazy dog']



# CountVectorizer

cv = CountVectorizer()



# Tokenize & build vocab

cv.fit(text)



# Summarize

print("Vocab Summary: ", cv.vocabulary_)



# Encode Text

vec = cv.transform(text)



# summarize encoded vector

print("Vector Shape: ",vec.shape)

print("Vector Type: ",type(vec))

print("Vector Array",vec.toarray())

# encode another document

text2 = ["every dog must have his day"]

vector = cv.transform(text2)

print(vector.toarray())
# Define text

text = ["The quick brown fox jumped over the lazy dog", "The dog", "The fox"]



# create the 

vectorizer = TfidfVectorizer()



# tokenize and build vocab

vectorizer.fit(text)



# summarize

print("Vocabulary Summary: ", vectorizer.vocabulary_)

print("Vector IDF:", vectorizer.idf_)



# encode document

vector = vectorizer.transform([text[0]])



# summarize encoded vector

print("Vector Shape: ",vector.shape)

print("Vector Array: ",vector.toarray())