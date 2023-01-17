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
import pandas as pd
import numpy as np

import  matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df.info()
df = df.drop(['EmployeeNumber'],axis = 1)
df = df.drop(['Over18'],axis = 1)
df = df.drop(['EmployeeCount'],axis=1)
df = df.drop(['StandardHours'],axis=1)
df.columns
df['Attrition'].value_counts().plot(kind='bar', color="blue", alpha=.65)
plt.title("Attrition Breakdown")
plt.figure(figsize=(18,15))
sns.heatmap(df.corr(),annot=False)
pd.crosstab(df['Gender'],df['Attrition']).plot(kind='bar')
plt.title('Attrition with respect to Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency of Attrition')
pd.crosstab(df['BusinessTravel'],df['Attrition']).plot(kind='bar')
plt.title('Attrition with respect to BusinessTravel')
plt.xlabel('BusinessTravel')
plt.ylabel('Frequency of Attrition')
plt.xticks(rotation=40)
pd.crosstab(df['Department'],df['Attrition']).plot(kind='bar', stacked=True)
plt.title('Attrition with respect to Department')
plt.xlabel('Department')
plt.ylabel('Frequency of Attrition')
plt.xticks(rotation=40)

pd.crosstab(df['EducationField'],df['Attrition']).plot(kind='bar',stacked=False)
plt.title('Attrition with respect to EducationField')
plt.xlabel('EducationField')
plt.ylabel('Frequency of Attrition')

pd.crosstab(df['JobRole'],df['Attrition']).plot(kind='bar', stacked=False)
plt.title('Attrition with respect to JobRole')
plt.xlabel('JobRole')
plt.ylabel('Frequency of Attrition')
pd.crosstab(df['MaritalStatus'],df['Attrition']).plot(kind='bar', stacked=False)
plt.title('Attrition with respect to MaritalStatus')
plt.xlabel('MaritalStatus')
plt.ylabel('Frequency of Attrition')
df['Gender'] = df['Gender'].map({'Female':0, 'Male':1}).astype(int)
df['BusinessTravel'] = df['BusinessTravel'].map({'Travel_Rarely':2, 'Travel_Frequently':1, 'Non-Travel':0}).astype(int)
df['OverTime'] = df['OverTime'].map({'Yes':0, 'No':1}).astype(int)
dummy1 = pd.get_dummies(df['EducationField'])
dummy2 = pd.get_dummies(df['JobRole'])
dummy3 = pd.get_dummies(df['MaritalStatus'])
dummy4 = pd.get_dummies(df['Department'])
df=pd.concat([df,dummy1,dummy2,dummy3,dummy4],axis=1)
df=df.drop(['EducationField','JobRole','MaritalStatus','Department'],axis=1)
df['Attrition'] = df['Attrition'].map({'Yes':0, 'No':1}).astype(int)
X=df.drop(['Attrition'],axis=1)
Y=df['Attrition']

X.shape
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_trainval_org, X_test_org, y_trainval, y_test = train_test_split(X,Y, random_state = 2)

# split train+validation set into training and validation sets
X_train_org, X_valid_org, y_train, y_valid = train_test_split(X_trainval_org, y_trainval, random_state=1)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train_org)
X_valid = scaler.fit_transform(X_valid_org)
X_trainval = scaler.fit_transform(X_trainval_org)
X_test = scaler.transform(X_test_org)

print("Size of training set: {}   size of validation set: {}   size of test set:"
      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

%matplotlib inline
train_score_array = []
valid_score_array = []

best_score = 0

for k in range(1,10):
    knn_clf = KNeighborsClassifier(k)
    knn_clf.fit(X_train, y_train)
    train_score_array.append(knn_clf.score(X_train, y_train))
    score = knn_clf.score(X_valid, y_valid)
    valid_score_array.append(score)
    if score > best_score:
            best_score = score
            best_parameters = {'K': k}
            best_K = k

x_axis = range(1,10)
plt.plot(x_axis, train_score_array, c = 'g', label = 'Train Score')
plt.plot(x_axis, valid_score_array, c = 'b', label = 'Validation Score')
plt.legend()
plt.xlabel('k')
plt.ylabel('MSE')

print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
knn_grid = KNeighborsClassifier(best_K)

scores = cross_val_score(knn_grid, X_trainval, y_trainval, cv =10, scoring = 'accuracy')
print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.2f}".format(scores.mean()))
k_range = list(range(1, 11))

param_grid = dict(n_neighbors=k_range)

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, return_train_score=True)

grid_search.fit(X_trainval, y_trainval)

df = pd.DataFrame(grid_search.cv_results_)
%matplotlib inline
x_axis = range(1,11)
plt.plot(x_axis, df.mean_train_score, c = 'g', label = 'Train Score')
plt.plot(x_axis, df.mean_test_score, c = 'b', label = 'Validation Score')
plt.legend()
plt.xlabel('k')
plt.ylabel('CV Score')

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report

 
pred_knn = grid_search.predict(X_test)
print(metrics.accuracy_score(y_test,pred_knn))

confusion = confusion_matrix(y_test, pred_knn)
print("Confusion matrix:\n{}".format(confusion))

print(classification_report(y_test,pred_knn))
from sklearn.metrics import precision_recall_fscore_support as score

precision,recall,fscore,support=score(y_test,pred_knn)

print ('Recall    : {}'.format(recall[0]))
print ('F1-Score    : {}'.format(fscore[0]))
Classification_Scores={}

Classification_Scores.update({'KNN Classification':[metrics.accuracy_score(y_test,pred_knn),recall[0],fscore[0]]})
columns = ['Classifier','Best Parameters','Accuracy_Score','Recall of 0']
clf_model_para = pd.DataFrame(columns=columns)

clf_model_para=clf_model_para.append({'Classifier':'KNN Classification',
                                      'Best Parameters':grid_search.best_params_,
                                      'Accuracy_Score':metrics.accuracy_score(y_test,pred_knn),
                                      'Recall of 0':recall[0]},ignore_index=True)
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

from sklearn.linear_model import LogisticRegression

c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_score_l1 = []
train_score_l2 = []
valid_score_l1 = []
valid_score_l2 = []

best_score = 0
l1 = 'l1'
l2 = 'l2'

for c in c_range:
    log_l1 = LogisticRegression(penalty = 'l1', C = c,solver='liblinear')
    log_l2 = LogisticRegression(penalty = 'l2', C = c,solver='lbfgs')
    
    log_l1.fit(X_train, y_train)
    log_l2.fit(X_train, y_train)
    
    train_score_l1.append(log_l1.score(X_train, y_train))
    train_score_l2.append(log_l2.score(X_train, y_train))
    
    score = log_l1.score(X_valid, y_valid)
    valid_score_l1.append(score)
    if score > best_score:
            best_score = score
            best_parameters = {'C': c , 'penalty': l1}
            best_C = c
            best_Penalty = 'l1'
    
    score = log_l2.score(X_valid, y_valid)
    valid_score_l2.append(score)
    if score > best_score:
            best_score = score
            best_parameters = {'C': c , 'penalty' : l2}
            best_C = c
            best_Penalty = 'l2'
    
plt.subplot(1,2,1)
plt.plot(c_range, train_score_l1, label = 'Train score, penalty = l1')
plt.plot(c_range, valid_score_l1, label = 'Test score, penalty = l1')
plt.xscale('log')
plt.legend()
plt.subplot(1,2,2)
plt.plot(c_range, train_score_l2, label = 'Train score, penalty = l2')
plt.plot(c_range, valid_score_l2, label = 'Test score, penalty = l2')
plt.xscale('log')
plt.legend()

print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))
log_grid = LogisticRegression(penalty = best_Penalty, C = best_C)

scores = cross_val_score(log_grid, X_trainval, y_trainval, cv =10, scoring = 'accuracy')
print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.2f}".format(scores.mean()))
param_grid = {'penalty': ['l1','l2'],
             'C':  [0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=10, return_train_score=True)

grid_search.fit(X_trainval, y_trainval)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
pred_log = grid_search.predict(X_test)
print(metrics.accuracy_score(y_test,pred_log))

confusion = confusion_matrix(y_test, pred_log)
print("Confusion matrix:\n{}".format(confusion))

print(classification_report(y_test,pred_log))
from sklearn.metrics import precision_recall_fscore_support as score

precision,recall,fscore,support=score(y_test,pred_log)

print ('Recall    : {}'.format(recall[0]))
print ('F1Score    : {}'.format(fscore[0]))
Classification_Scores.update({'Logistic Classification':[metrics.accuracy_score(y_test,pred_log),recall[0],fscore[0]]})
clf_model_para=clf_model_para.append({'Classifier':'Logistic Classification',
                                      'Best Parameters':grid_search.best_params_,
                                      'Accuracy_Score':metrics.accuracy_score(y_test,pred_log),
                                      'Recall of 0':recall[0]},ignore_index=True)
from sklearn.svm import LinearSVC

train_score_list = []
valid_score_list = []

best_score = 0

for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    linear_svc = LinearSVC(C=C, max_iter=10000)
    linear_svc.fit(X_train,y_train)
    train_score_list.append(linear_svc.score(X_train,y_train))
    score = linear_svc.score(X_valid, y_valid)
    valid_score_list.append(score)
    if score > best_score:
        best_score = score
        best_parameters = {'C' : C}
        best_C = C

x_range = [0.001, 0.01, 0.1, 1, 10, 100]
plt.plot(x_range, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_range, valid_score_list, c = 'b', label = 'Validation Score')
plt.xscale('log')
plt.legend(loc = 3)
plt.xlabel('Regularization parameter')

print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))
linear_svc_grid = LinearSVC(C = best_C, max_iter=10000)
scores = cross_val_score(linear_svc_grid, X_trainval, y_trainval, cv = 10, scoring = 'accuracy')
print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.2f}".format(scores.mean()))
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LinearSVC(max_iter=10000), param_grid, cv=10, return_train_score=True)

grid_search.fit(X_trainval, y_trainval)

df = pd.DataFrame(grid_search.cv_results_)
%matplotlib inline
plt.plot(x_range, df.mean_train_score, c = 'g', label = 'Train Score')
plt.plot(x_range, df.mean_test_score, c = 'b', label = 'Validation Score')
plt.xscale('log')
plt.legend(loc = 3)
plt.xlabel('Regularization Parameter')

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
pred_linear_svc = grid_search.predict(X_test)
print(metrics.accuracy_score(y_test,pred_linear_svc))

confusion = confusion_matrix(y_test, pred_linear_svc)
print("Confusion matrix:\n{}".format(confusion))

print(classification_report(y_test,pred_linear_svc))
from sklearn.metrics import precision_recall_fscore_support as score

precision,recall,fscore,support=score(y_test,pred_linear_svc)

print ('Recall    : {}'.format(recall[0]))
print ('F1Score    : {}'.format(fscore[0]))
Classification_Scores.update({'Linear_SVC':[metrics.accuracy_score(y_test,pred_linear_svc),recall[0],fscore[0]]})
clf_model_para=clf_model_para.append({'Classifier':'Linear_SVC',
                                      'Best Parameters':grid_search.best_params_,
                                      'Accuracy_Score':metrics.accuracy_score(y_test,pred_linear_svc),
                                      'Recall of 0':recall[0]},ignore_index=True)
from sklearn.svm import SVC

train_score_list = []
valid_score_list = []

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svc_rbf = SVC(kernel='rbf', gamma=gamma, C=C)
        svc_rbf.fit(X_train,y_train)
        train_score_list.append(svc_rbf.score(X_train,y_train))
        score = svc_rbf.score(X_valid, y_valid)
        valid_score_list.append(score)
        if score > best_score:
            best_score = score
            best_parameters = {'gamma': gamma , 'C' : C}
            best_Gamma = gamma
            best_C = C

print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))
svc_rbf_grid = SVC(kernel='rbf', gamma = best_Gamma, C = best_C)

scores = cross_val_score(svc_rbf_grid, X_trainval, y_trainval, cv =10, scoring = 'accuracy')
print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.2f}".format(scores.mean()))
param_grid = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
             'C': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=10, return_train_score=True)

grid_search.fit(X_trainval, y_trainval)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
pred_rbf = grid_search.predict(X_test)
print(metrics.accuracy_score(y_test,pred_rbf))

confusion = confusion_matrix(y_test, pred_rbf)
print("Confusion matrix:\n{}".format(confusion))

print(classification_report(y_test,pred_rbf))
from sklearn.metrics import precision_recall_fscore_support as score

precision,recall,fscore,support=score(y_test,pred_rbf)

print ('Recall    : {}'.format(recall[0]))
print ('F1-Score    : {}'.format(fscore[0]))
Classification_Scores.update({'SVC RBF Kernel':[metrics.accuracy_score(y_test,pred_rbf),recall[0],fscore[0]]})
clf_model_para=clf_model_para.append({'Classifier':'SVC RBF Kernel',
                                      'Best Parameters':grid_search.best_params_,
                                      'Accuracy_Score':metrics.accuracy_score(y_test,pred_rbf),
                                      'Recall of 0':recall[0]},ignore_index=True)
train_score_list = []
valid_score_list = []

best_score = 0

for degree in range(1,5):
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
            svc_poly = SVC(kernel='poly', degree = degree, C=C, gamma = gamma)
            svc_poly.fit(X_train,y_train)
            train_score_list.append(svc_poly.score(X_train,y_train))
            score = svc_poly.score(X_valid, y_valid)
            valid_score_list.append(score)
            if score > best_score:
                best_score = score
                best_parameters = {'degree': degree , 'C' : C, 'gamma' : gamma}
                best_Degree = degree
                best_C = C
                best_gamma = gamma

print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))
svc_poly_grid = SVC(kernel='poly',degree = best_Degree, C=best_C, gamma = best_Gamma)

scores = cross_val_score(svc_poly_grid, X_trainval, y_trainval, cv =10, scoring = 'accuracy')
print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.2f}".format(scores.mean()))
param_grid = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
             'C': [0.001, 0.01, 0.1, 1, 10, 100],
             'degree': [1,2,3,4,5]}

grid_search = GridSearchCV(SVC(kernel='poly'), param_grid, cv=10, return_train_score=True)

grid_search.fit(X_trainval, y_trainval)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
pred_poly = grid_search.predict(X_test)
print(metrics.accuracy_score(y_test,pred_poly))

confusion = confusion_matrix(y_test, pred_poly)
print("Confusion matrix:\n{}".format(confusion))

print(classification_report(y_test,pred_poly))
from sklearn.metrics import precision_recall_fscore_support as score

precision,recall,fscore,support=score(y_test,pred_poly)

print ('Recall    : {}'.format(recall[0]))
print ('FScore    : {}'.format(fscore[0]))
Classification_Scores.update({'SVC Poly Kernel':[metrics.accuracy_score(y_test,pred_poly),recall[0],fscore[0]]})
clf_model_para=clf_model_para.append({'Classifier':'SVC Poly Kernel',
                                      'Best Parameters':grid_search.best_params_,
                                      'Accuracy_Score':metrics.accuracy_score(y_test,pred_poly),
                                      'Recall of 0':recall[0]},ignore_index=True)
from sklearn.svm import SVC
train_score_list = []
valid_score_list = []

best_score = 0

for degree in range(1,5):
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
            svc_poly = SVC(kernel='linear', degree = degree, C=C, gamma = gamma)
            svc_poly.fit(X_train,y_train)
            train_score_list.append(svc_poly.score(X_train,y_train))
            score = svc_poly.score(X_valid, y_valid)
            valid_score_list.append(score)
            if score > best_score:
                best_score = score
                best_parameters = {'degree': degree , 'C' : C, 'gamma' : gamma}
                best_Degree = degree
                best_C = C
                best_gamma = gamma

print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

svc_poly_grid = SVC(kernel='linear',degree = best_Degree, C=best_C, gamma = best_gamma)

scores = cross_val_score(svc_poly_grid, X_trainval, y_trainval, cv =10, scoring = 'accuracy')
print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.2f}".format(scores.mean()))
param_grid = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
             'C': [0.001, 0.01, 0.1, 1, 10, 100],
             'degree': [1,2,3,4,5]}

grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=10, return_train_score=True)

grid_search.fit(X_trainval, y_trainval)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report 
pred_linear = grid_search.predict(X_test)
print(metrics.accuracy_score(y_test,pred_linear))

confusion = confusion_matrix(y_test, pred_linear)
print("Confusion matrix:\n{}".format(confusion))

print(classification_report(y_test,pred_linear))
Classification_Scores.update({'SVC Poly Linear':[metrics.accuracy_score(y_test,pred_linear),recall[0],fscore[0]]})
clf_model_para=clf_model_para.append({'Classifier':'SVC Poly Linear',
                                      'Best Parameters':grid_search.best_params_,
                                      'Accuracy_Score':metrics.accuracy_score(y_test,pred_linear),
                                      'Recall of 0':recall[0]},ignore_index=True)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_trainval, y_trainval)
print("Accuracy on training set: {:.3f}".format(dtree.score(X_trainval, y_trainval)))
print("Accuracy on test set: {:.3f}".format(dtree.score(X_test, y_test)))
dtree_cv = DecisionTreeClassifier()
scores = cross_val_score(dtree_cv, X_trainval, y_trainval, cv = 10, scoring = 'accuracy' )
print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.2f}".format(scores.mean()))
pred_tree = dtree.predict(X_test)
print(metrics.accuracy_score(y_test,pred_tree))

confusion = confusion_matrix(y_test, pred_tree)
print("Confusion matrix:\n{}".format(confusion))

print(classification_report(y_test,pred_tree))
from sklearn.metrics import precision_recall_fscore_support as score

precision,recall,fscore,support=score(y_test,pred_tree)

print ('Recall    : {}'.format(recall[0]))
print ('F1Score    : {}'.format(fscore[0]))
Classification_Scores.update({'Decison Tree':[metrics.accuracy_score(y_test,pred_tree),recall[0],fscore[0]]})
clf_model_para=clf_model_para.append({'Classifier':'Decision Tree',
                                      'Best Parameters':' ',
                                      'Accuracy_Score':metrics.accuracy_score(y_test,pred_tree),
                                      'Recall of 0':recall[0]},ignore_index=True)
Classification_Scores=pd.DataFrame(Classification_Scores)
Classification_Scores.rename({0:'Accuracy_Score',1:'Recall',2:'F1-Score'})
plt.figure(figsize=(56,5))

Classification_Scores.plot.bar(figsize=(20,10))