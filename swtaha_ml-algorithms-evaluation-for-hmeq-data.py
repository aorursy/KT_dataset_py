#import the packages



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input")) #list the files in the input directory

df=pd.read_csv('../input/hmeq.csv') #import the dataset

columnNames = pd.Series(df.columns.values) # to check the columns/variables/features present in our data set
df.head(5) # to see the first 5 rows of dataset

df.shape #to look at the shape of the dataset
#descriptive statistics

description= df.describe(include='all') # to get the basic summary of all the numeric columns and frequency distribution of all the categorical columns.

description
data_types=df.dtypes #to print data types for each variable

data_types
MissingData=df.isnull().sum().rename_axis('Variables').reset_index(name='Missing Values') # the isnull() returns 1 if the value is null

MissingData
#dropping rows that have missing data

df.dropna(axis=0, how='any', inplace=True)

df

#Frequency distribution of target variable "BAD" and visualizing the target variable

df["BAD"].value_counts().plot.bar(title='BAD')
#visualizing the categorical variable REASON

REASON_count= df["REASON"].value_counts().rename_axis('REASON').reset_index(name='Total Count')

df["REASON"].value_counts().plot.bar(title='REASON')

#visualizing the categorical variable JOB

JOB_count= df["JOB"].value_counts().rename_axis('JOB').reset_index(name='Total Count')

df["JOB"].value_counts().plot.bar(title='JOB')
# visualizing numeric variables using seaborn

f, axes = plt.subplots(3, 3, figsize=(25,25))

sns.distplot( df["LOAN"] , color="skyblue", ax=axes[0, 0])

sns.distplot( df["DEBTINC"] , color="olive", ax=axes[0, 1])

sns.distplot( df["MORTDUE"] , color="orange", ax=axes[0, 2])

sns.distplot( df["YOJ"] , color="yellow", ax=axes[1, 0])

sns.distplot( df["VALUE"] , color="pink", ax=axes[1, 1])

sns.distplot( df["DELINQ"] , color="red", ax=axes[1, 2])

sns.distplot( df["DEROG"] , color="green", ax=axes[2, 0])

sns.distplot( df["CLAGE"] , color="gold", ax=axes[2, 1])

sns.distplot( df["CLNO"] , color="teal", ax=axes[2, 2])


JOB=pd.crosstab(df['JOB'],df['BAD'])

JOB.div(JOB.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='JOB vs BAD', figsize=(4,4))

REASON=pd.crosstab(df['REASON'],df['BAD'])

REASON.div(REASON.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='REASON vs BAD', figsize=(4,4))
# visualizing numeric variables using seaborn

f, axes = plt.subplots(3, 3, figsize=(25,25))

sns.distplot( df["LOAN"] , color="skyblue", ax=axes[0, 0])

sns.distplot( df["DEBTINC"] , color="olive", ax=axes[0, 1])

sns.distplot( df["MORTDUE"] , color="orange", ax=axes[0, 2])

sns.distplot( df["YOJ"] , color="yellow", ax=axes[1, 0])

sns.distplot( df["VALUE"] , color="pink", ax=axes[1, 1])

sns.distplot( df["DELINQ"] , color="red", ax=axes[1, 2])

sns.distplot( df["DEROG"] , color="green", ax=axes[2, 0])

sns.distplot( df["CLAGE"] , color="gold", ax=axes[2, 1])

sns.distplot( df["CLNO"] , color="teal", ax=axes[2, 2])


dfWithBin = df.copy()

bins=[0,15000,25000,90000] 

group=['Low','Average','High'] 

dfWithBin['LOAN_bin']=pd.cut(df['LOAN'],bins,labels=group)

LOAN_bin=pd.crosstab(dfWithBin['LOAN_bin'],dfWithBin['BAD'])

LOAN_bin.div(LOAN_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,title='Realtionship between Amount of Loan requested and the target variable BAD')

plt.xlabel('LOAN')

P= plt.ylabel('Percentage')
bins=[0,47000,92000,400000] 

group=['Low','Average','High'] 

dfWithBin['MORTDUE_bin']=pd.cut(dfWithBin['MORTDUE'],bins,labels=group)

LOAN_bin=pd.crosstab(dfWithBin['MORTDUE_bin'],dfWithBin['BAD'])

LOAN_bin.div(LOAN_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,title='Realtionship between the Amount due on existing mortgage and the target variable BAD')

plt.xlabel('MORTDUE')

P= plt.ylabel('Percentage')
bins=[0,68000,120000,860000] 

group=['Low','Average','High'] 

dfWithBin['VALUE_bin']=pd.cut(dfWithBin['VALUE'],bins,labels=group)

LOAN_bin=pd.crosstab(dfWithBin['VALUE_bin'],dfWithBin['BAD'])

LOAN_bin.div(LOAN_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,title='Realtionship between the value of the current property and the target variable BAD')

plt.xlabel('VALUE')

P= plt.ylabel('Percentage')
bins=[0,3,15] 

group=['Low','High'] 

dfWithBin['DELINQ_bin']=pd.cut(dfWithBin['DELINQ'],bins,labels=group)

LOAN_bin=pd.crosstab(dfWithBin['DELINQ_bin'],dfWithBin['BAD'])

LOAN_bin.div(LOAN_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,title='Relationship of Number of Delinquent credit lines with the target variable')

plt.xlabel('DELINQ')

P= plt.ylabel('Percentage')
bins=[0,2,15] 

group=['Low','High'] 

dfWithBin['DEROG_bin']=pd.cut(dfWithBin['DEROG'],bins,labels=group)

LOAN_bin=pd.crosstab(dfWithBin['DEROG_bin'],dfWithBin['BAD'])

LOAN_bin.div(LOAN_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,title='Relationship of Number of major derogatory reports with the target variable')

plt.xlabel('DEROG')

P= plt.ylabel('Percentage')
bins=[0,120,230,1170] 

group=['Low','Average','High'] 

dfWithBin['CLAGE_bin']=pd.cut(dfWithBin['CLAGE'],bins,labels=group)

LOAN_bin=pd.crosstab(dfWithBin['CLAGE_bin'],dfWithBin['BAD'])

LOAN_bin.div(LOAN_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,title='Relationship  of Age of oldest tradeline in months with the target variable')

plt.xlabel('CLAGE')

P= plt.ylabel('Percentage')
bins=[0,40,204] 

group=['Low','High'] 

dfWithBin['DEBTINC_bin']=pd.cut(dfWithBin['DEBTINC'],bins,labels=group)

LOAN_bin=pd.crosstab(dfWithBin['DEBTINC_bin'],dfWithBin['BAD'])

LOAN_bin.div(LOAN_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,title='Debt to Income ratio realtionship with target variable')

plt.xlabel('DEBTINC')

P= plt.ylabel('Percentage')
#Create Correlation matrix

corr = df.corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(10,8))

#Generate Color Map

colormap = sns.diverging_palette(220, 10, as_cmap=True)

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

#Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

#Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

#show plot

plt.show()
#encoding

df=pd.get_dummies(df, columns=['REASON','JOB'])

df

# Extract independent and target variables

X = df.drop(['BAD'], axis=1)

y = df['BAD']
#Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = pd.DataFrame(sc_X.fit_transform(X), columns=X.columns)
#RFE with the logistic regression algorithm to select the top 4 features. 

#import classifier

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

model = LogisticRegression()

rfe = RFE(model, 4)

fit = rfe.fit(X, y)

no_of_features = fit.n_features_

support_features = fit.support_

ranking_features = fit.ranking_

print("Num Features: %d" % (no_of_features))

print("Selected Features: %s" % (support_features))

print("Feature Ranking: %s" % (ranking_features))

X_sub = X.iloc[:,support_features] #updated X with the top 4 features
#splitting the data into test and train for logistic regression

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_sub,y,random_state = 0) # default 25% test data

#import logistic regresiion model

from sklearn.linear_model import LogisticRegression

# create model (estimator) object

classifier = LogisticRegression()

# fit model to training data

classifier.fit(X_train,y_train)

#classifier performance on test set

classifier.score(X_test,y_test)

# make predictions

y_pred = classifier.predict(X_test)

y_score= classifier.predict_proba(X_test)


#import performance measure tools

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, classification_report

acs=accuracy_score(y_test,y_pred)

rs=recall_score(y_test,y_pred, average='macro') 

ps=precision_score(y_test,y_pred, average='macro') 

print("accuracy score : ",acs)

print("precision score : ",rs)

print("recall score : ",ps)

#print("Accuracy : %s" % "{0:.3%}".format(acs))

print(classification_report(y_test, y_pred))
import itertools

def plot_confusion_matrix(cm,classes=[0,1],title='Confusion matrix without normalization', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    fmt = 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# Compute confusion matrix

cm = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

print('Confusion matrix without normalization')

print(cm)

plt.figure()

plot_confusion_matrix(cm)

plt.show()

#ROC plot

from sklearn.metrics import roc_curve, auc

def plot_roc(y_test, y_score):

    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])



    plt.figure()

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title("ROC plot for loan defaulter prediction")

    plt.legend(loc="lower right")

    plt.show()

#Plot ROC

plot_roc(y_test, y_score)
# KFOLD

from statistics import mean, stdev



from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold, cross_val_score, cross_validate

logreg = LogisticRegression()

kf = KFold(n_splits=5,shuffle=True,random_state=1)

kf_scores = []

xmat = X_sub.as_matrix()

ymat = y.as_matrix()

for train_index, test_index in kf.split(xmat):

    X_train, y_train=xmat[train_index], ymat[train_index]

    logreg.fit(X_train, y_train)

    y_predicted=logreg.predict(X_test)

    kf_scores.append(accuracy_score(y_test, y_predicted))



print(kf_scores)
#import KNN classifier

from sklearn.neighbors import KNeighborsClassifier

model= KNeighborsClassifier()

#parameters of gridsearch for KNN:n_neighbors, weights

param_dict= {'n_neighbors':range(3,11,2),

             'weights':['uniform','distance'],

                       'p':[1,2,3,4,5]

                       }

#since this is binary classifier use odd for list of neighbours , with the range function we get k=3,5,7,9

#uniform means that every near neighbour will get the same weightage whether k =3,5,7 or 9

#p means the manhatten or eucledean or higher power distances

#grid search to find the best parameters

from sklearn.model_selection import GridSearchCV

best_model = GridSearchCV(model, param_dict, cv=5) 

best_model.fit(X_sub, y)

best_model.best_params_ #{'n_neighbors': 7, 'p': 1, 'weights': 'uniform'}

best_model.best_score_  #best_score will give the mean score of 5 cv's which is 0.9304399524375743

ypred = best_model.predict(X_sub)

y_score= best_model.predict_proba(X_sub)
# import Performance measure

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

acs=accuracy_score(y,ypred) 

rs=recall_score(y,ypred, average='macro') 

ps=precision_score(y,ypred, average='macro') 

print("accuracy score : ",acs)

print("precision score : ",rs)

print("recall score : ",ps)
# Compute confusion matrix for KNN

cm = confusion_matrix(y, ypred)

np.set_printoptions(precision=2)

print('Confusion matrix without normalization')

print(cm)

plt.figure()

plot_confusion_matrix(cm)

plt.show()
#Plot ROC for KNN

plot_roc(y,y_score)
#import SVM classifier

from sklearn.svm import SVC

model=SVC(probability=True)

param_dict = {'kernel':['rbf','poly'],'degree': [1,2,3], 'C':[0.5,0.75,1],'gamma': [0.01, 0.1, 1]}

#parameters for grid search for SVM are kernel,degree,gamma and C

from sklearn.model_selection import GridSearchCV

#best_model = GridSearchCV(model, param_dict, cv=5, scoring= 'precision') 

best_model = GridSearchCV(model, param_dict, cv=5) 

best_model.fit(X_sub, y)

best_model.best_params_ # {'C': 1, 'degree': 1, 'gamma': 0.1, 'kernel': 'rbf'}

best_model.best_score_ # 0.9316290130796671

ypred = best_model.predict(X_sub)

y_score= best_model.predict_proba(X_sub)
# import Performance measure for SVM

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

acs=accuracy_score(y,ypred) 

rs=recall_score(y,ypred, average='macro') 

ps=precision_score(y,ypred, average='macro')

print("accuracy score : ",acs)

print("precision score : ",rs)

print("recall score : ",ps)
#compute Confusion Matrix for SVC

cm = confusion_matrix(y, ypred)

np.set_printoptions(precision=2)

print('Confusion matrix without normalization')

print(cm)

plt.figure()

plot_confusion_matrix(cm)

plt.show()
#Plot ROC for SVC

plot_roc(y,y_score)
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

param_dict={'min_samples_split' : range(10,500,20),'max_depth': np.arange(3, 10)}

clf=GridSearchCV(model,param_dict, cv=5)

best_model.fit(X, y)

best_model.best_params_ # 

best_model.best_score_ # 0.9316290130796671

ypred = best_model.predict(X)

y_score= best_model.predict_proba(X)

# import Performance measure

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

acs=accuracy_score(y,ypred) 

rs=recall_score(y,ypred, average='macro')  

ps=precision_score(y,ypred, average='macro') 

print("accuracy score : ",acs)

print("precision score : ",rs)

print("recall score : ",ps)
# Compute confusion matrix for Decision Tree

cm = confusion_matrix(y, ypred)

np.set_printoptions(precision=2)

print('Confusion matrix without normalization')

print(cm)

plt.figure()

plot_confusion_matrix(cm)
#Plot ROC for Decision Tree

plot_roc(y,y_score)
#feature importance of decision tree

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0) #

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

model.fit(X_train, y_train) # train the data first for calculating the feature importance

featimp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

print (featimp)

featimp.plot(kind='bar', title='Feature Importance of Decision Tree Model')

#import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_jobs=-1)

param_dict = { 'n_estimators':[5,10,15],

               'max_depth':[50,60,70],

               'criterion': ['gini','entropy']

              }

#parameters for grid search for Randomforest are n_estimators, max depth and criterion

from sklearn.model_selection import GridSearchCV

best_model = GridSearchCV(model, param_dict, cv=5) 

best_model.fit(X, y)

best_model.best_params_ #{'criterion': 'gini', 'max_depth': 60, 'n_estimators': 10}

best_model.best_score_  #0.9200356718192628

ypred = best_model.predict(X)

y_score= best_model.predict_proba(X)
# import Performance measure

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

acs=accuracy_score(y,ypred) 

rs=recall_score(y,ypred, average='macro') 

ps=precision_score(y,ypred, average='macro')

print("accuracy score : ",acs)

print("precision score : ",rs)

print("recall score : ",ps)
#compute confusion matrix for Random Forest

cm = confusion_matrix(y, ypred)

np.set_printoptions(precision=2)

print('Confusion matrix without normalization')

print(cm)

plt.figure()

plot_confusion_matrix(cm)
#Plot ROC for Random Forest

plot_roc(y,y_score)
#feature importance of random forest

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0) #

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_jobs=-1)

model.fit(X_train, y_train) # train the data first for calculating the feature importance

featimp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

print (featimp)

featimp.plot(kind='bar', figsize=(15,7.5),title='Feature Importance of Random Forest Model')