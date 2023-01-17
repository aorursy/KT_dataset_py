# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, warnings, gc

warnings.filterwarnings("ignore")



# SKLearn Classification Algorithm

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# SKLearn Generic

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, validation_curve, KFold, cross_val_score

from sklearn.preprocessing import LabelEncoder, StandardScaler



# Tabulation

from tabulate import tabulate

url = '../input/all-datasets-for-practicing-ml/Class/Class_Abalone.csv'

data = pd.read_csv(url, header='infer')
# Total Records

print("Total Records: ", data.shape[0])
#Check for null/missing values

print("Is there missing data? - ", data.empty)
# Records per sex

data.groupby("Sex").size()
#Stat Summary

data.describe().transpose()
# Instantiating Label Encoder

encoder = LabelEncoder()



#Encoding Sex column

data['Sex'] = encoder.fit_transform(data['Sex'])





#Feature & Target Selection

columns = data.columns

target = ['Sex']

features = columns[1:]



X = data[features]

y = data[target]





# Dataset Split 

''' Training = 90% & Validation = 10%  '''

test_size = 0.1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True) 





#Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_val = sc.transform(X_val)
# -- Building Model List --

models = []



models.append(('GaussianNB', GaussianNB()))       # Gaussian Naive Bayes

models.append(('RandomForest', RandomForestClassifier(verbose=0, random_state=1, max_features=8))) #Random Forest

models.append(('KNN',KNeighborsClassifier()))  #KNN

models.append(('DecisionTree',DecisionTreeClassifier(max_features=8, random_state=1)))  #Decision Tree

models.append(('LinearDiscriminant',LinearDiscriminantAnalysis()))  #Linear Discriminant Analysis

# Voting Classifier

vc = VotingClassifier(estimators=[('gnb', GaussianNB()),('rf', RandomForestClassifier(verbose=0, random_state=1, max_features=8)),

                                  ('knn', KNeighborsClassifier()), ('dt', DecisionTreeClassifier(max_features=8, random_state=1)), 

                                  ('lda', LinearDiscriminantAnalysis())], voting='hard')



#Fit

vc.fit(X_train, y_train)



# Appending the Voting Classifier to Model List

models.append(('VotingClassifier',vc)) 



for label, model in models:

    kfold = KFold(n_splits=10, random_state=None, shuffle=False)

    cross_val = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    print("Accuracy Score: %0.2f -- [%s]" % (cross_val.mean(), label))
# Train & Predict (top 5 records)

gnb_pred = GaussianNB().fit(X_train, y_train).predict(X_val[:5,])

rf_pred = RandomForestClassifier(verbose=0, random_state=1, max_features=8).fit(X_train, y_train).predict(X_val[:5,])

knn_pred = KNeighborsClassifier().fit(X_train, y_train).predict(X_val[:5,])

dt_pred = DecisionTreeClassifier(max_features=8, random_state=1).fit(X_train, y_train).predict(X_val[:5,])

lda_pred = LinearDiscriminantAnalysis().fit(X_train, y_train).predict(X_val[:5,])

vc_pred = vc.predict(X_val[:5,])

# Tabulating the classifier predictions of first 5 records 

tab_pred = []



for idx,(a,b,c,d,e,f) in enumerate(zip(gnb_pred,rf_pred,knn_pred,dt_pred,lda_pred,vc_pred)):

    tab_pred.append([idx+1,a,b,c,d,e,f])



print("Classifier Predictions: \n",tabulate(tab_pred, headers=["Record","GaussianNB","RandomF","KNN","DTree","LDA","Voting"], tablefmt='pretty'))
# -- Building Model List --  

models = []



models.append(('GaussianNB', GaussianNB()))       # Gaussian Naive Bayes

models.append(('RandomForest', RandomForestClassifier(verbose=0, random_state=1, max_features=8))) #Random Forest

models.append(('KNN',KNeighborsClassifier()))  #KNN

models.append(('DecisionTree',DecisionTreeClassifier(max_features=8, random_state=1)))  #Decision Tree

models.append(('LinearDiscriminant',LinearDiscriminantAnalysis()))  #Linear Discriminant Analysis
# Voting Classifier

vc = VotingClassifier(estimators=[('gnb', GaussianNB()),('rf', RandomForestClassifier(verbose=0, random_state=1, max_features=8)),

                                  ('knn', KNeighborsClassifier()), ('dt', DecisionTreeClassifier(max_features=8, random_state=1)), 

                                  ('lda', LinearDiscriminantAnalysis())], voting='soft')



#Fit

vc.fit(X_train, y_train)



# Appending the Voting Classifier to Model List

models.append(('VotingClassifier',vc)) 



for label, model in models:

    kfold = KFold(n_splits=10, random_state=None, shuffle=False)

    cross_val = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    print("Accuracy Score: %0.2f -- [%s]" % (cross_val.mean(), label))
# Train & Predict (top 5 records)

gnb_pred = GaussianNB().fit(X_train, y_train).predict(X_val[:5,])

rf_pred = RandomForestClassifier(verbose=0, random_state=1, max_features=8).fit(X_train, y_train).predict(X_val[:5,])

knn_pred = KNeighborsClassifier().fit(X_train, y_train).predict(X_val[:5,])

dt_pred = DecisionTreeClassifier(max_features=8, random_state=1).fit(X_train, y_train).predict(X_val[:5,])

lda_pred = LinearDiscriminantAnalysis().fit(X_train, y_train).predict(X_val[:5,])

vc_pred = vc.predict(X_val[:5,])
# Tabulating the classifier predictions of first 5 records 

tab_pred = []



for idx,(a,b,c,d,e,f) in enumerate(zip(gnb_pred,rf_pred,knn_pred,dt_pred,lda_pred,vc_pred)):

    tab_pred.append([idx+1,a,b,c,d,e,f])



print("Classifier Predictions: \n",tabulate(tab_pred, headers=["Record","GaussianNB","RandomF","KNN","DTree","LDA","Voting"], tablefmt='pretty'))