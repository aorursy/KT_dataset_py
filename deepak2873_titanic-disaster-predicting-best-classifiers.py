## 3.1. Importing essential libraries

## ----------------------------------

# allow plots to appear directly in the notebook

%matplotlib inline



import itertools

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import preprocessing



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier #RandomForestRegressor

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier



from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import accuracy_score



#from sklearn.externals import joblib

import csv



## 3.2. Importing datasets

## -----------------------



train_data = pd.read_csv('../input/titanic-test-data-disaster/train.csv')

test_data = pd.read_csv('../input/titanic-test-data-disaster/test.csv')



#train_data._get_numeric_data()

train_data.head()
# Lets see the features and columns present in train_data and test_data

train_data.columns
# Lets see the features and columns present in train_data and test_data

test_data.columns
train_data.corr()["Survived"]
# Data Cleaning: convert the feature 'Sex' -male and female to '0' and '1' respectively.

train_data['Sex'] = train_data['Sex'].replace(['male','female'],[0,1])
train_data.head()
train_data.corr()["Survived"]
survivor_count = train_data['Survived'].value_counts()

print (survivor_count)
survivor_count[:2].plot(kind='bar',color='red')

## class 0 - deceased

## class 1 - Survived
print("Accuracy- survived percent in predict train data: %.3f%%" % ((342/(549 + 342))*100.0))
survivor_sex_count = train_data.groupby(['Survived','Sex']).size()  # .describe()



print (survivor_sex_count)
survivor_sex_count[:10].plot(kind='bar',color='violet', title='Survivor vs Gender')

#train_data = train_data[['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

train_data = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

#train_data = train_data[['Pclass', 'Sex', 'Parch', 'Fare', 'Survived']]

#train_data['Sex'] = train_data['Sex'].replace(['male','female'],[0,1])  # 

train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())

train_data["Fare"] = train_data["Fare"].fillna(train_data["Fare"].median())



test_data = test_data[['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

#test_data = test_data[['PassengerId','Pclass', 'Sex',  'Parch', 'Fare']]

test_data['Sex'] = test_data['Sex'].replace(['male','female'],[0,1])

test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median())

test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())
train_data.corr()["Survived"]
X = train_data.drop('Survived', axis=1)

y = train_data['Survived']

X_test = test_data.drop('PassengerId', axis=1)



#print (X)
svc = SVC()

knc = KNeighborsClassifier()

#mnb = MultinomialNB()

dtc = DecisionTreeClassifier()

lrc = LogisticRegression()

rfc = RandomForestClassifier()

abc = AdaBoostClassifier()

bc = BaggingClassifier()

etc = ExtraTreesClassifier()



classifiers = {'SVC' : svc,'KN' : knc, 'DT': dtc, 'LR': lrc,  'ABC': abc, 'BgC': bc, 'ETC': etc,'RFC':rfc}

#classifiers = {'SVC' : svc,'KN' : knc, 'NB': mnb, 'DT': dtc, 'LR': lrc,  'ABC': abc, 'BgC': bc, 'ETC': etc}
def hyper_parameters(k):

    if k == 'SVC':

        parameter = {'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid' ]}

    elif k == 'KN':

        parameter = {'kneighborsclassifier__n_neighbors': [3,5,10]}

    #elif k == 'NB':

    #    parameter = { 'multinomialnb__alpha': [1.0, 2.0, 5.0]}

    elif k == 'DT':

        parameter = { 'decisiontreeclassifier__criterion': ['gini','entropy'],

                     'decisiontreeclassifier__max_features' : ['auto', 'sqrt', 'log2'], 

                     'decisiontreeclassifier__max_depth': [None, 5, 3, 1]}

    elif k == 'LR':

        parameter = {'logisticregression__random_state': [None, 1, 5, 10]}

    elif k == 'RFC':

        parameter = { 'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],

                   'randomforestclassifier__max_depth': [None, 5, 3, 1]}

    elif k == 'ABC':

        parameter = {'adaboostclassifier__n_estimators': [3,5,10]}

    elif k == 'BgC':

        parameter = {'baggingclassifier__n_estimators': [5,10, 20], 'baggingclassifier__random_state': [None, 10,100]}

    elif k == 'ETC':

        parameter =  {'extratreesclassifier__random_state': [None,1,5],

                      'extratreesclassifier__criterion': ['gini','entropy'],

                     'extratreesclassifier__max_features': ['auto', 'sqrt', 'log2'], 

                      'extratreesclassifier__n_estimators':[3,5, 7]}

        

        

    return parameter

train_clf = []

acc_scores = []

pred_scores = []

for k,clf in classifiers.items():

    #print (k,"   :   ",clf)

    pipe = make_pipeline(preprocessing.StandardScaler(), clf)

    #sorted(pipe.get_params().keys())

    #print (pipe.get_params())

    

    hyperparameters = hyper_parameters(k)

    trainedclfs = GridSearchCV(pipe, hyperparameters, cv=5)

    

    # Fit and tune model

    trainedclfs.fit(X, y)

    print("_" * 20)

    print (k,' - ',trainedclfs.best_params_)

    pred = trainedclfs.predict(X)

    train_clf.append ((k, trainedclfs))

    acc_scores.append ((k, accuracy_score(y,pred)))

    pred_scores.append((k, [accuracy_score(y,pred)]))



#df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])

#df.sort_values(by ='Score')

#print (acc_scores)

#print (pred_scores)

labels  = ['Clf','Score']



accuracy_scoredf = pd.DataFrame.from_records(acc_scores, columns=labels)

#df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])



sorted_accuracy = accuracy_scoredf.sort_values(by ='Score',ascending=False)

sorted_accuracy
#

#sorted_accuracy = sorted_accuracy * 100

sns.barplot(x=sorted_accuracy.Clf, y = sorted_accuracy.Score, data = sorted_accuracy, color = 'salmon')



plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('classifiers')

plt.ylabel('Accuracy Score (%)')
from sklearn.metrics import classification_report, confusion_matrix



for i in range(0, 8):

    for j in range(0, 1):

        if train_clf[i][j] =='ETC':

            #print(train_clf[i][j+1])

            selectedclf =  train_clf[i][j+1]

            pred_on_train = selectedclf.predict(X)

            

            if train_clf[i][j] =='ETC':

                pred_on_test  = selectedclf.predict(X_test)   



            print(pred_on_test.size)

                        

            # Compute confusion matrix

            print("Confusion Matrix of : " , train_clf[i][j],'\n')

            cnf_matrix = confusion_matrix(train_data['Survived'], pred_on_train)

                        

            

            print(cnf_matrix)

            

            sns.heatmap(cnf_matrix,annot=True,fmt="d")

            print('\n')

            tn, fp, fn, tp = confusion_matrix(train_data['Survived'],pred_on_train).ravel()

            print ("True Positive  - Actually Survived and model predicts it correctly [0,0]      : ", tp)

            print ("False Negative - Actually Survived and model predicts it Not-Survived [1,0]   : ", fn)

            print ("False Positive - Actually Not-Survived and model predicts it Survived [0,1]   : ", fp)

            print ("True Negative  - Actually Not-Survived and model predicts it correctly [1,1]  : ", tn)

            print('\n')

            # Compute Classification Report

            print("Classification Report of : ", train_clf[i][j], '\n')

            cls_report = classification_report(train_data['Survived'], pred_on_train)

            print(cls_report)

            
id=(test_data['PassengerId'])

pred_on_test.size
## Save the the output to "submission.csv" file

#submission = pd.DataFrame({'PassengerId' : id, 'Survived': pred_on_test})

#submission.to_csv('..\submission.csv', index=False)