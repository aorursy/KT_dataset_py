# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import time

from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_score

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as make_pipeline_imb

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score, auc, f1_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/creditcard.csv")
#view the columns of the dataset

data.columns
#view the first 5 samples of the dataset

data.head()
#view the last 4 samples of the dataset

data.tail(4)
#pick any 3 samples of the dataset

data.sample(3)
data.info()
#no missing values

max(data.isnull().sum())
#have a view of the features in the dataset

data.columns
fraud = data[data.Class==1]

non_fraud = data[data.Class==0]



#fraction of frauds

frac_of_fraud = len(fraud)/len(data)

#fraction of non-frauds

frac_of_non_fraud = len(non_fraud)/len(data)



print('Percentage of Frauds: {}%'.format(round(frac_of_fraud*100, 2)))

print('Percentage of Non-Frauds: {}%'.format(round(frac_of_non_fraud*100, 2)))
#visualize the imbalance with a bar chart

plt.title('Distribution of Frauds', fontdict={'size' : 16, 'color':'brown'})

sns.countplot(x='Class', data=data)

labels = ['Non-Fraud', 'Fraud']   #to label the plot

vals = [0, 1]   #to put the labels right



plt.xticks(vals, labels)

plt.xlabel('Class', fontdict={'size' : 14, 'color' : 'green'})

plt.ylabel('Number of transactions', fontdict={'size' : 12, 'color':'green'})
fig, ax = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)



ax1 = sns.distplot(data['Time'], color='brown', ax=ax[0])

ax2 = sns.distplot(data['Amount'], ax=ax[1])



ax1.set_title('Distribution of Transaction Time')

ax2.set_title('Distribution of Transaction Amount')
#skewness of the Time column

print('The Transaction Time has a skewness of {}'.format(round(data.Time.skew(), 4)))



#skewness of the Amount column

print('The Transaction Amount has a skewness of {}'.format(round(data.Amount.skew(), 4)))



#skewness of the Class column

print('The Class - Target Variable - has a skewness of {}'.format(round(data.Class.skew(), 4)))
#import the RobustScaler estimator class

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()



#apply RobustScaler to the Time and the Amount columns

data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])



#view the dataset to see the new look of the Time and Amount columns

data.head(5)
X = data.drop('Class', axis=1)  #independent variables

y = data['Class']   #dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, 

                                                    random_state=0)
#importing the classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
models = {

    'Logistic Regression' : LogisticRegression(), 

    'Naive Bayes' : GaussianNB(),

     #'Support Vector Classifier' : SVC(),  computationally expensive to run on the whole dataset

    #'Decision Tree Classifier' : DecisionTreeClassifier()   computationally expensive

    

}
for name, model in models.items():

    t0 = time.time()  #start time

    model.fit(X_train, y_train)

    accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()

    t1 = time.time()  #stop time

    print(name, 'Score: {}%'.format(round(accuracy, 2)*100))

    print('Computation Time: {}s\n'.format(round(t1-t0, 2)))

    print('*'*80)

    print()
#since I do not know whether the classes are distributed in any unique pattern, I will shuffle the whole dataset

data = data.sample(frac=1, random_state=42)



#pick out the fraud and the non-fraud samples from the shuffled dataset

fraud = data[data.Class==1]

non_fraud = data[data.Class==0]



#print out the number of samples in fraud and non_fraud

print('Before Undersampling: ')

print('Number of Fraudulent Transactions: {}'.format(len(fraud)))

print('Number of Non-Fraudulent Transactions: {}'.format(len(non_fraud)))

print()



#making the non_fraud transactions(majority class) equal to the fraud transactions(minority class)

non_fraud = non_fraud.sample(frac=1)

non_fraud = non_fraud[:len(fraud)]



#the non_fraud transactions are now equal to the fraud transaction --- let's visualize

print('After Undersampling: ')

print('Number of Fraudulent Transactions: {}'.format(len(fraud)))

print('Number of Non-Fraudulent Transactions: {}'.format(len(non_fraud)))





#now join the fraud dataset to the non_fraud dataset

sampled_data = pd.concat([fraud, non_fraud], axis=0)

#shuffle the sampled_data to allow for random distribution of classes in the dataset

sampled_data = sampled_data.sample(frac=1, random_state=42)
fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)



ax1 = sns.countplot(x='Class', data=data, ax=ax[0])

ax1.set_title('Distribution of Classes before Undersampling', color='brown')



ax2 = sns.countplot(x='Class', data=sampled_data, ax=ax[1])

ax2.set_title('Distribution of Classes after Undersampling', color='red')
#a pie chart will also indicate a 50/50 ratio of fraud to non_fraud

patches, texts, autotexts = plt.pie(

    x=[len(fraud), len(non_fraud)], 

    labels=['Fraud', 'Non-Fraud'],

    explode = [0.012, 0.012],

    shadow=True,

    autopct='%.1f%%',

    radius=1.2,

    startangle=30

)



for text in texts:

    text.set_color('#22AA11')

    text.set_size(14)

for autotext in autotexts:

    autotext.set_color('red')
X_undersampled = sampled_data.drop('Class', axis=1)

y_undersampled = sampled_data['Class']





#splitting the dataset

X_train_undersampled, X_test_undersampled, y_train_undersampled, y_test_undersampled = train_test_split(X_undersampled, 

                                                                                                        y_undersampled, 

                                                                                                        stratify=y_undersampled, 

                                                                                                        test_size=0.25, 

                                                                                                        random_state=0)
#use the cross validation technique - StratifiedShuffleSplit - to perform GridSearch and to calculate the train and test scores

cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=0)





#using the Gaussian Naive Bayes on the dataset

bayes = GaussianNB()

bayes.fit(X_train_undersampled, y_train_undersampled)

bayes_score = cross_val_score(bayes, X_train_undersampled, y_train_undersampled, cv=cv).mean()

bayes_predictions = bayes.predict(X_test_undersampled)

bayes_precision_score = precision_score(y_test_undersampled, bayes_predictions)

bayes_recall_score = recall_score(y_test_undersampled, bayes_predictions)

bayes_auc = roc_auc_score(y_test_undersampled, bayes_predictions)











#parameters to search for using GridSearchCV



#Logistic Regression

logReg_params = {'penalty' : ['l1', 'l2'], 'C' : [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.5, 1.7, 2.0]}



#Support Vector Classifier

svc_params = {'C' : [0.1, 1, 10], 'gamma' : [1, 0.1, 0.01, 0.001], 'kernel' : ['linear', 'rbf'] }



#Decision Tree Classifier

dtree_params = {'criterion' : ['gini', 'entropy'], 'max_depth' : [3, 4, 5]}









#GridSearch on Logistic Regression

logReg_grid = GridSearchCV(LogisticRegression(), logReg_params, refit=True, verbose=0, cv=cv, scoring='accuracy')

logReg_grid.fit(X_train_undersampled, y_train_undersampled)

logReg = logReg_grid.best_estimator_



logReg_score = cross_val_score(logReg, X_train_undersampled, y_train_undersampled, cv=cv).mean()

logReg_predictions = logReg.predict(X_test_undersampled)

logReg_precision_score = precision_score(y_test_undersampled, logReg_predictions)

logReg_recall_score = recall_score(y_test_undersampled, logReg_predictions)

logReg_auc = roc_auc_score(y_test_undersampled, logReg_predictions)





#GridSearch on Support Vector Classifier

svc_grid = GridSearchCV(SVC(), svc_params, refit=True, verbose=0, cv=cv, scoring='accuracy')

svc_grid.fit(X_train_undersampled, y_train_undersampled)

svc = svc_grid.best_estimator_



svc_score = cross_val_score(svc, X_train_undersampled, y_train_undersampled, cv=cv).mean()

svc_predictions = svc.predict(X_test_undersampled)

svc_precision_score = precision_score(y_test_undersampled, svc_predictions)

svc_recall_score = recall_score(y_test_undersampled, svc_predictions)

svc_auc = roc_auc_score(y_test_undersampled, svc_predictions)





#GridSearch on Decision Tree Classifier

dtree_grid = GridSearchCV(DecisionTreeClassifier(), dtree_params, refit=True, verbose=0, cv=cv, scoring='accuracy')

dtree_grid.fit(X_train_undersampled, y_train_undersampled)

dtree = dtree_grid.best_estimator_



dtree_score = cross_val_score(dtree, X_train_undersampled, y_train_undersampled, cv=cv).mean()

dtree_predictions = dtree.predict(X_test_undersampled)

dtree_precision_score = precision_score(y_test_undersampled, dtree_predictions)

dtree_recall_score = recall_score(y_test_undersampled, dtree_predictions)

dtree_auc = roc_auc_score(y_test_undersampled, dtree_predictions)
model_names = ['Logistic Regression', 'Gaussian Naive Bayes', 'Support Vector Classifier', 'Decision Tree Classifier']
for name in model_names:

    if name == 'Logistic Regression':

        print(name, 'Scores: \n')

        print('Accuracy: {}'.format(round(logReg_score, 2)))

        print('Precision: {}'.format(round(logReg_precision_score, 2)))

        print('Recall: {}'.format(round(logReg_recall_score, 2)))

        print('AUC: {}\n'.format(round(logReg_auc, 2)))

        print('*'*90)

    elif name == 'Gaussian Naive Bayes':

        print(name, 'Scores: \n')

        print('Accuracy: {}'.format(round(bayes_score, 2)))

        print('Precision: {}'.format(round(bayes_precision_score, 2)))

        print('Recall: {}'.format(round(bayes_recall_score, 2)))

        print('AUC: {}\n'.format(round(bayes_auc, 2)))

        print('*'*90)

    elif name == 'Support Vector Classifier':

        print(name, 'Scores: \n')

        print('Accuracy: {}'.format(round(svc_score, 2)))

        print('Precision: {}'.format(round(svc_precision_score, 2)))

        print('Recall: {}'.format(round(svc_recall_score, 2)))

        print('AUC: {}\n'.format(round(svc_auc, 2)))

        print('*'*90)

    elif name == 'Decision Tree Classifier':

        print(name, 'Scores: \n')

        print('Accuracy: {}'.format(round(dtree_score, 2)))

        print('Precision: {}'.format(round(dtree_precision_score, 2)))

        print('Recall: {}'.format(round(dtree_recall_score, 2)))

        print('AUC: {}\n'.format(round(dtree_auc, 2)))

        print('*'*90)
print('Random Undersampling Before Data Split')

print('.'*60)

print()



logReg_conf_matrix = confusion_matrix(y_test_undersampled, logReg_predictions)

bayes_conf_matrix = confusion_matrix(y_test_undersampled, bayes_predictions)

svc_conf_matrix = confusion_matrix(y_test_undersampled, svc_predictions)

dtree_conf_matrix = confusion_matrix(y_test_undersampled, dtree_predictions)





fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)



fig.subplots_adjust(hspace=0.6, wspace=0.4)  #adjust the spaces between the subplots

tick_labels = ['non_fraud', 'fraud']  #labels for the xticks and yticks





#Logistic Regression Confusion Matrix

ax1 = sns.heatmap(logReg_conf_matrix, cbar=False, yticklabels=tick_labels, xticklabels=tick_labels, annot=True, ax=ax[0][0])

ax1.set_title('Logistic Regression Confusion Matrix', color='red')

ax1.set_ylabel('Actual Labels', size=8)

ax1.set_xlabel('Predicted Labels', size=8)





#Gaussian Naive Bayes Confusion Matrix

ax2 = sns.heatmap(bayes_conf_matrix, cbar=False, yticklabels=tick_labels, xticklabels=tick_labels, annot=True, ax=ax[0][1])

ax2.set_title('Gaussian Naive Bayes Confusion Matrix', color='red')

ax2.set_ylabel('Actual Labels', size=8)

ax2.set_xlabel('Predicted Labels', size=8)





#Support Vector Classifier Confusion Matrix

ax3 = sns.heatmap(svc_conf_matrix, cbar=False, yticklabels=tick_labels, xticklabels=tick_labels, annot=True, ax=ax[1][0])

ax3.set_title('Support Vector Classifier Confusion Matrix', color='red')

ax3.set_ylabel('Actual Labels', size=8)

ax3.set_xlabel('Predicted Labels', size=8)





#Decision Tree Confusion Matrix

ax4 = sns.heatmap(dtree_conf_matrix, cbar=False, yticklabels=tick_labels, xticklabels=tick_labels, annot=True, ax=ax[1][1])

ax4.set_title('Decision Tree Classifier Confusion Matrix', color='red')

ax4.set_ylabel('Actual Labels', size=8)

ax4.set_xlabel('Predicted Labels', size=8)

cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=0)



accuracy = []

precision = []

recall = []

auc = []







#remember X_train and y_train are from the original data

#join X_train and y_train; join X_test and y_test

#then undersample only the resulting dataframe of X_train and y_train; and then test using the dataframe resulting from X_test and y_test

train_data = pd.concat([X_train, y_train], axis=1)

test_data = pd.concat([X_test, y_test], axis=1)





#pick out fraud and non_fraud from the train_data

fraud = train_data[train_data.Class==1]

non_fraud = train_data[train_data.Class==0]



#make fraud equal to non_fraud by picking out random samples from non_fraud and making it equal to fraud

non_fraud = non_fraud.sample(n=len(fraud))



#number of fraud and non_fraud

print('Number of Fraudulent Transactions: {}'.format(len(fraud)))

print('Number of Non-Fraudulent Transactions: {}'.format(len(non_fraud)))



#concatenate fraud and non_fraud and call the dataframe undersampled_train_data

undersampled_train_data = pd.concat([fraud, non_fraud], axis=0)

undersampled_train_data = undersampled_train_data.sample(frac=1)  #resample your data to allow for a kind of random/even distribution



#we will use the whole undersampled_train_data to train our model, and test it using the test_data that isn't undersampled!

train_X = undersampled_train_data.drop('Class', axis=1)

train_y = undersampled_train_data['Class']



#split the test_data

test_X = test_data.drop('Class', axis=1)

test_y = test_data['Class']





#use gridsearch to search for parameters

logReg_params = {'penalty' : ['l1', 'l2'], 'C' : [0.4, 0.6, 0.8, 1.0, 1.3, 1.5, 1.7, 2.0]}



log_grid = GridSearchCV(LogisticRegression(random_state=0), logReg_params, refit=True, verbose=0)

log_grid.fit(train_X, train_y)   #training with the undersampled data

logReg = log_grid.best_estimator_

#testing with the original dataset which was not undersampled

predict = log_grid.predict(test_X)





#scores

logReg_score = cross_val_score(logReg, train_X, train_y, cv=cv).mean()



logReg_precision_score = precision_score(test_y, predict)

logReg_recall_score = recall_score(test_y, predict)

logReg_auc = roc_auc_score(test_y, predict)
print('Logistic Regression Results for Random Undersampling After Data Split\n')

print('*'*78)

print()

print('Accuracy: {}'.format(round(logReg_score, 2)))

print('Precision: {}'.format(round(logReg_precision_score, 2)))

print('Recall: {}'.format(round(logReg_recall_score, 2)))

print('AUC: {}'.format(round(logReg_auc, 2)))
print('Random Undersampling After Data Split')

print('.'*60)

print()



ax1 = sns.heatmap(confusion_matrix(test_y, predict), cbar=False, yticklabels=tick_labels, xticklabels=tick_labels, annot=True)

ax1.set_title('Logistic Regression Confusion Matrix', color='red')

ax1.set_ylabel('Actual Labels', size=8)

ax1.set_xlabel('Predicted Labels', size=8)
#function to display the results for the results of NearMiss and SMOTE before cv and during cv

def show_metrics(title, accuracy, precision, recall, f1, auc):

    print(title, 'Results:\n')

    print('Accuracy: {}'.format(round(np.mean(accuracy), 2)))

    print('Precision: {}'.format(round(np.mean(precision), 2)))

    print('Recall: {}'.format(round(np.mean(recall), 2)))

    print('F1-Score: {}'.format(round(np.mean(f1), 2)))

    print('AUC: {}'.format(round(np.mean(auc), 2)))

    print()

    print('*'*80)

    print()
#cross validation technique to use during NearMiss

cv = StratifiedShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
nearmiss_before_cv_accuracy = []         #accuracy score

nearmiss_before_cv_precision = []        #precision score

nearmiss_before_cv_recall = []           #recall

nearmiss_before_cv_f1 = []               #f1 score

nearmiss_before_cv_auc = []              #auc --- area under curve



nearmiss_before_cv_conf_matrices = []    #list to append the confusion matrices obtained during CV





print('Distribution of Target Variable in Original Data: {}\n'.format(Counter(y)))   #distribution of fraud and non_fraud in original dataset

print('Distribution of Target Variable in Training Set of Original Data: {}\n'.format(Counter(y_train)))   #distribution of fraud and non_fraud in the training part of the original dataset





#remember that X_train and y_train are a split from the original dataset

X_nearmiss, y_nearmiss = NearMiss(random_state=0).fit_sample(X_train, y_train)   #applying NearMiss before CV, WRONG!!!



print('Distribution of Target Variable in Training Set of Original Data after NearMiss: {}\n'.format(Counter(y_nearmiss)))  #distribution of fraud and non_fraud must be equal in the training part of the original dataset after NearMiss is applied



for train, test in cv.split(X_nearmiss, y_nearmiss):   #CV begins here!

    #pipeline for imbalanced data -- from imblearn.pipeline import make_pipeline as make_make_pipeline_imb

    pipeline = make_pipeline_imb(LogisticRegression(random_state=42)) 

    nearmiss_before_cv_model = pipeline.fit(X_nearmiss[train], y_nearmiss[train])

    nearmiss_before_cv_predictions = nearmiss_before_cv_model.predict(X_nearmiss[test])

    

    #scoring metrics

    nearmiss_before_cv_accuracy.append(pipeline.score(X_nearmiss[test], y_nearmiss[test]))

    nearmiss_before_cv_precision.append(precision_score(y_nearmiss[test], nearmiss_before_cv_predictions))

    nearmiss_before_cv_recall.append(recall_score(y_nearmiss[test], nearmiss_before_cv_predictions))

    nearmiss_before_cv_f1.append(f1_score(y_nearmiss[test], nearmiss_before_cv_predictions))

    nearmiss_before_cv_auc.append(roc_auc_score(y_nearmiss[test], nearmiss_before_cv_predictions))

    

    #confusion matrices --- since the no. of splits is 10, I have 10 different confusion matrices

    nearmiss_before_cv_conf_matrices.append(confusion_matrix(y_nearmiss[test], nearmiss_before_cv_predictions))
nearmiss_during_cv_accuracy = []

nearmiss_during_cv_precision = []

nearmiss_during_cv_recall = []

nearmiss_during_cv_f1 = []

nearmiss_during_cv_auc = []





nearmiss_during_cv_conf_matrices = []



#in order to get the values of the indices obtained during cv easily, I'll convert X_train and y_train into numpy arrays; I can also do this by using the pandas' 'iloc' method

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values





for train, test in cv.split(X_train, y_train):   #CV begins here!

    pipeline = make_pipeline_imb(NearMiss(random_state=0), LogisticRegression(random_state=42)) #NearMiss during CV, CORRECT!!!

    nearmiss_during_cv_model = pipeline.fit(X_train[train], y_train[train])

    nearmiss_during_cv_predictions = nearmiss_during_cv_model.predict(X_train[test])

    

    #scoring metrics

    nearmiss_during_cv_accuracy.append(pipeline.score(X_train[test], y_train[test]))

    nearmiss_during_cv_precision.append(precision_score(y_train[test], nearmiss_during_cv_predictions))

    nearmiss_during_cv_recall.append(recall_score(y_train[test], nearmiss_during_cv_predictions))

    nearmiss_during_cv_f1.append(f1_score(y_train[test], nearmiss_during_cv_predictions))

    nearmiss_during_cv_auc.append(roc_auc_score(y_train[test], nearmiss_during_cv_predictions))

    

    #confusion matrices --- since the no. of splits is 10, I must have 10 confusion matrices

    nearmiss_during_cv_conf_matrices.append(confusion_matrix(y_train[test], nearmiss_during_cv_predictions))
print('RESULTS for the NearMiss Algorithm')

print('.'*80)

print()



show_metrics('NearMiss Before Cross Validation', nearmiss_before_cv_accuracy, nearmiss_before_cv_precision, nearmiss_before_cv_recall, nearmiss_before_cv_f1, nearmiss_before_cv_auc)

show_metrics('NearMiss During Cross Validation', nearmiss_during_cv_accuracy, nearmiss_during_cv_precision, nearmiss_during_cv_recall, nearmiss_during_cv_f1, nearmiss_during_cv_auc)
fig, ax = plt.subplots(figsize=(20, 3), nrows=1, ncols=5)



i = 0

print('NearMiss Before CV Confusion Matrices')

print('.'*60)

#the first 5 confusion matrices of the 10

for nearmiss_before_cv_conf_matrix in nearmiss_before_cv_conf_matrices[:5]:

    ax0 = sns.heatmap(nearmiss_before_cv_conf_matrix, cbar=False,xticklabels=tick_labels, yticklabels=tick_labels, ax=ax[i], annot=True, annot_kws={'size' : 15})

    #ax0.set_ylabel('Actual Labels')

    #ax0.set_xlabel('Predicted Labels')

    i = i + 1
fig, ax = plt.subplots(figsize=(20, 3), nrows=1, ncols=5)



k = 0

print('NearMiss During CV Confusion Matrices')

print('.'*60)

#the first 5 confusion matrices of the 10

for nearmiss_before_cv_conf_matrix in nearmiss_during_cv_conf_matrices[:5]:

    ax1 = sns.heatmap(nearmiss_before_cv_conf_matrix, cbar=False, xticklabels=tick_labels, yticklabels=tick_labels, ax=ax[k], annot=True, annot_kws={'size' : 15})

    #ax1.set_ylabel('Actual Labels')

    #ax1.set_xlabel('Predicted Labels')

    k= k + 1
#cross validation technique to use during SMOTE

kf = StratifiedKFold(n_splits=3, shuffle=False, random_state=0)
smote_before_cv_accuracy = []         

smote_before_cv_precision = []        

smote_before_cv_recall = []           

smote_before_cv_f1 = []               

smote_before_cv_auc = []              



smote_before_cv_conf_matrices = []  





print('Distribution of Target Variable in Original Data: {}\n'.format(Counter(y)))   #distribution of fraud and non_fraud in original dataset

print('Distribution of Target Variable in Training Set of Original Data: {}\n'.format(Counter(y_train)))   #distribution of fraud and non_fraud in the training part of the original dataset





smote_X, smote_y = SMOTE(random_state=0).fit_sample(X_train, y_train)   #applying SMOTE before CV, WRONG!!!



print('Distribution of Target Variable in Training Set of Original Data after NearMiss: {}\n'.format(Counter(smote_y)))  #distribution of fraud and non_fraud must be equal in the training part of the original dataset after SMOTE is applied





for train, test in kf.split(smote_X, smote_y):   #CV begins here!

    pipeline = make_pipeline_imb(LogisticRegression(random_state=42)) 

    smote_before_cv_model = pipeline.fit(smote_X[train], smote_y[train])

    smote_before_cv_predictions = smote_before_cv_model.predict(smote_X[test])

    

    #classification metrics

    smote_before_cv_accuracy.append(pipeline.score(smote_X[test], smote_y[test]))

    smote_before_cv_precision.append(precision_score(smote_y[test], smote_before_cv_predictions))

    smote_before_cv_recall.append(recall_score(smote_y[test], smote_before_cv_predictions))

    smote_before_cv_f1.append(f1_score(smote_y[test], smote_before_cv_predictions))

    smote_before_cv_auc.append(roc_auc_score(smote_y[test], smote_before_cv_predictions))

    

    #confusion matrices --- since the no. of splits is 10, I have 10 different confusion matrices

    smote_before_cv_conf_matrices.append(confusion_matrix(smote_y[test], smote_before_cv_predictions))
smote_during_cv_accuracy = []

smote_during_cv_precision = []

smote_during_cv_recall = []

smote_during_cv_f1 = []

smote_during_cv_auc = []





smote_during_cv_conf_matrices = []



#no need to convert X_train, X_test, y_train and y_test to numpy arrays, as I have done that before

X_train = X_train

X_test = X_test

y_train = y_train

y_test = y_test





for train, test in kf.split(X_train, y_train):   #CV begins here!

    pipeline = make_pipeline_imb(SMOTE(random_state=0), LogisticRegression(random_state=42)) #SMOTE during CV, CORRECT!!!

    smote_during_cv_model = pipeline.fit(X_train[train], y_train[train])

    smote_during_cv_predictions = smote_during_cv_model.predict(X_train[test])

    

    #scoring metrics

    smote_during_cv_accuracy.append(pipeline.score(X_train[test], y_train[test]))

    smote_during_cv_precision.append(precision_score(y_train[test], smote_during_cv_predictions))

    smote_during_cv_recall.append(recall_score(y_train[test], smote_during_cv_predictions))

    smote_during_cv_f1.append(f1_score(y_train[test], smote_during_cv_predictions))

    smote_during_cv_auc.append(roc_auc_score(y_train[test], smote_during_cv_predictions))

    

    

    #confusion matrices --- since the no. of splits is 10, I must have 10 confusion matrices

    smote_during_cv_conf_matrices.append(confusion_matrix(y_train[test], smote_during_cv_predictions))
print('RESULTS for SMOTE')

print('.'*80)

print()



show_metrics('SMOTE Before Cross Validation', smote_before_cv_accuracy, smote_before_cv_precision, smote_before_cv_recall, smote_before_cv_f1, smote_before_cv_auc)

show_metrics('SMOTE During Cross Validation', smote_during_cv_accuracy, smote_during_cv_precision, smote_during_cv_recall, smote_during_cv_f1, smote_during_cv_auc)
fig, ax = plt.subplots(figsize=(9, 3), nrows=1, ncols=3)



x = 0

print('NearMiss Before CV Confusion Matrices')

print('.'*60)

#the first 5 confusion matrices of the 10

for smote_before_cv_conf_matrix in smote_before_cv_conf_matrices[:5]:

    ax0 = sns.heatmap(smote_before_cv_conf_matrix, cbar=False, xticklabels=tick_labels, yticklabels=tick_labels, ax=ax[x], annot=True, annot_kws={'size' : 10})

    #ax0.set_ylabel('Actual Labels')

    #ax0.set_xlabel('Predicted Labels')

    x = x + 1
fig, ax = plt.subplots(figsize=(9, 3), nrows=1, ncols=3)



y = 0

print('NearMiss Before CV Confusion Matrices')

print('.'*60)

#the first 5 confusion matrices of the 10

for smote_during_cv_conf_matrix in smote_during_cv_conf_matrices[:5]:

    ax1 = sns.heatmap(nearmiss_before_cv_conf_matrix, cbar=False, xticklabels=tick_labels, yticklabels=tick_labels, ax=ax[y], annot=True, annot_kws={'size' : 10})

    #ax0.set_ylabel('Actual Labels')

    #ax0.set_xlabel('Predicted Labels')

    y = y + 1