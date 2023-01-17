#working with data
import pandas as pd
import numpy as np

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

## Scikit-learn features various classification, regression and clustering algorithms
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, classification_report,f1_score

## Scaling
from sklearn.preprocessing import StandardScaler

## Algo
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import warnings
warnings.filterwarnings('ignore')

#loading Data
Data = pd.read_csv('../input/parkinsondata/Data-Parkinsons')
Data.head()
#fetch all columns
Data.columns
#checking data Type of each attributes
Data.dtypes
shape_data=Data.shape
print('Data set contains "{x}" number of rows and "{y}" number of columns columns'.format(x=shape_data[0],y=shape_data[1]))
#checking for Null Values
Data.isnull().sum()
sns.heatmap(Data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#overview of data
Data.describe().transpose()
#A Skewness value of 0 in the output denotes a symmetrical distribution
#A negative Skewness value in the output denotes tail is larger towrds left hand side of data so we can say left skewed
#A Positive Skewness value in the output denotes tail is larger towrds Right hand side of data so we can say Right skewed
Data.skew()
#Univariate analysis of Fundamental frequency
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.distplot(Data['MDVP:Fo(Hz)'],bins=30,ax=axes[0])
sns.distplot(Data['MDVP:Fhi(Hz)'],bins=30,ax=axes[1])
sns.distplot(Data['MDVP:Flo(Hz)'],bins=30,ax=axes[2])
axes[0].set_title('Average vocal fundamental frequency')
axes[1].set_title('Maximum vocal fundamental frequency')
axes[2].set_title('Minimum vocal fundamental frequency')
#Univariate analysis of measures of variation in fundamental frequency
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
sns.distplot(Data['MDVP:Jitter(%)'],bins=30,ax=axes[0,0],color='green')
sns.distplot(Data['MDVP:Jitter(Abs)'],bins=30,ax=axes[0,1],color='green')
sns.distplot(Data['MDVP:RAP'],bins=30,ax=axes[0,2],color='green')

sns.distplot(Data['MDVP:PPQ'],bins=30,ax=axes[1,0],color='green')
sns.distplot(Data['Jitter:DDP'],bins=30,ax=axes[1,1],color='green')
#sns.distplot(Data['Shimmer:DDA'],bins=30,ax=axes[1,2])
fig.tight_layout()
#Univariate analysis of  variation in amplitude
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
sns.distplot(Data['MDVP:Shimmer'],bins=30,ax=axes[0,0],color='orange')
sns.distplot(Data['MDVP:Shimmer(dB)'],bins=30,ax=axes[0,1],color='orange')
sns.distplot(Data['Shimmer:APQ3'],bins=30,ax=axes[0,2],color='orange')

sns.distplot(Data['Shimmer:APQ5'],bins=30,ax=axes[1,0],color='orange')
sns.distplot(Data['MDVP:APQ'],bins=30,ax=axes[1,1],color='orange')
sns.distplot(Data['Shimmer:DDA'],bins=30,ax=axes[1,2],color='orange')

#analysis for measures of ratio of noise to tonal components in the voice
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
sns.distplot(Data['NHR'],bins=30,ax=axes[0],color='red')
sns.distplot(Data['HNR'],bins=30,ax=axes[1],color='red')

#analysis for wo nonlinear dynamical complexity measures
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
sns.distplot(Data['RPDE'],bins=30,ax=axes[0],color='purple')
sns.distplot(Data['D2'],bins=30,ax=axes[1],color='purple')
#Univariate analysis of nonlinear measures of fundamental frequency variation 
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.distplot(Data['spread1'],bins=30,ax=axes[0])
sns.distplot(Data['spread2'],bins=30,ax=axes[1])
sns.distplot(Data['PPE'],bins=30,ax=axes[2])
#Variation of traget variables
sns.countplot(x=Data['status'])
corr = Data.corr()
corr
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 3.5})
plt.figure(figsize=(18,7))
# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
#variation of Spread1 with Target Variable
sns.kdeplot(Data[Data.status == 0]['spread1'], shade=False,)
sns.kdeplot(Data[Data.status == 1]['spread1'], shade=True)
plt.title("Variation of Spread1 with Status")
#variation of HNR with Target Variable, will not consider MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA as they are highly correlatedwith NHR 
sns.boxplot(x='status',y='HNR',data=Data)
#variation of  Maximum, Minimum vocal fundamental frequency
fig, ax = plt.subplots(1,2,figsize=(14,4))
sns.kdeplot(Data[Data.status == 0]['MDVP:Flo(Hz)'], shade=False,ax=ax[0])
sns.kdeplot(Data[Data.status == 1]['MDVP:Flo(Hz)'], shade=True,ax=ax[0])

sns.kdeplot(Data[Data.status == 0]['MDVP:Fhi(Hz)'], shade=False,ax=ax[1])
sns.kdeplot(Data[Data.status == 1]['MDVP:Fhi(Hz)'], shade=True,color='r',ax=ax[1])
#variation of MDVP:Jitter(%) with Target Variable, will not consider Jitter(Abs),MDVP:RAP,MDVP:PPQ,NHR as they are highly correlatedwith NHR 
sns.boxplot(x='status',y='MDVP:Jitter(%)',data=Data)
#variation of Spread1 with Target Variable
sns.kdeplot(Data[Data.status == 0]['MDVP:Jitter(%)'], shade=False,)
sns.kdeplot(Data[Data.status == 1]['MDVP:Jitter(%)'], shade=True)
plt.title("Variation of MDVP:Jitter(%) with Status")
#Split the data into training and test set in the ratio of 70:30 respectively
X = Data.drop(['status','name'],axis=1)
y = Data['status']

# split data into train subset and test subset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# checking the dimensions of the train & test subset
# to print dimension of train set
print(X_train.shape)
# to print dimension of test set
print(X_test.shape)
#checking the variance 
#high variance means fearure does not affect the target variable
X_train.var()
#dropping correlated values which are have either more then 80% or less then -80%
X_train.drop(['MDVP:Shimmer','MDVP:Jitter(%)','HNR'],axis=1,inplace=True)
X_test.drop(['MDVP:Shimmer','MDVP:Jitter(%)','HNR'],axis=1,inplace=True)
#since there is lots of variety in the units of features let's scale it
scaler=StandardScaler().fit(X_train)
scaler_x_train=scaler.transform(X_train)

scaler=StandardScaler().fit(X_test)
scaler_x_test=scaler.transform(X_test)

# Train and Fit model
model = LogisticRegression(random_state=0)
model.fit(scaler_x_train, y_train)
#predict the Personal Loan Values
y_Logit_pred = model.predict(scaler_x_test)
y_Logit_pred
# Let's measure the accuracy of this model's prediction
print("confusion_matrix")
print(confusion_matrix(y_test,y_Logit_pred))
# And some other metrics for Test
cr=classification_report(y_test, y_Logit_pred, digits=2)
print(cr)
# creating odd list of K for KNN
myList = list(range(3,40,2))

# empty list that will hold accuracy scores
ac_scores = []

# perform accuracy metrics for values from 3,5....29
for k in myList:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(scaler_x_train, y_train)
    # predict the response
    y_pred = knn.predict(scaler_x_test)
    # evaluate F1 Score
    scores = f1_score(y_test, y_pred)
    ac_scores.append(scores)

# changing to misclassification error
MSE = [1 - x for x in ac_scores]

# determining best k
optimal_k = myList[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
# instantiate learning model (k = 29)
knn = KNeighborsClassifier(n_neighbors = 29, weights = 'uniform', metric='euclidean')
# fitting the model
knn.fit(scaler_x_train, y_train)

# predict the response
y_Knn_pred = knn.predict(scaler_x_test)

# Let's measure the accuracy of this model's prediction
print("confusion_matrix")
print(confusion_matrix(y_test,y_Knn_pred))

# evaluate Model Score
print(classification_report(y_test, y_Knn_pred, digits=2))
clf = SVC(gamma=0.05, C=3,random_state=0)
clf.fit(scaler_x_train , y_train)

# predict the response
prediction_SVC = clf.predict(scaler_x_test)

# Let's measure the accuracy of this model's prediction
print("confusion_matrix")
print(confusion_matrix(y_test,prediction_SVC))

# evaluate Model Score
print(classification_report(y_test, prediction_SVC, digits=2))
#Using K fold to check how my algorighm varies throughout my data if we split it in 10 equal bins
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('K-NN', KNeighborsClassifier(n_neighbors = 29, weights = 'uniform', metric='euclidean')))
models.append(('SVM', SVC(gamma=0.05, C=3)))

# evaluate each model
results = []
names = []
scoring = 'f1'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=101)
	cv_results = model_selection.cross_val_score(model, scaler_x_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	print("Name = %s , Mean F1-Score = %f, SD F1-Score = %f" % (name, cv_results.mean(), cv_results.std()))
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.plot(results[0],label='Logistic')
plt.plot(results[1],label='KNN')
plt.plot(results[2],label='SVC')
plt.legend()
plt.show()
#Stacking the idea of stacking is to learn several different weak learners
# and combine them by training a meta-model to output predictions based on the multiple predictions
# returned by these weak models. So, we need to define two things in order to build our stacking model:
# the L learners we want to fit and the meta-model that combines them.

# defining level hetrogenious model
level0 = list()
level0.append(('lr', LogisticRegression()))
level0.append(('knn', KNeighborsClassifier(n_neighbors = 29, weights = 'uniform', metric='euclidean')))
level0.append(('cart', DecisionTreeClassifier()))
level0.append(('svm', SVC(gamma=0.05, C=3)))
level0.append(('bayes', GaussianNB()))

# define meta learner model
level1 = SVC(gamma=0.05, C=3)

# define the stacking ensemble with cross validation of 5
Stack_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

# predict the response
Stack_model.fit(scaler_x_train, y_train)
prediction_Stack = Stack_model.predict(scaler_x_test)

# Let's measure the accuracy of this model's prediction
print("confusion_matrix")
print(confusion_matrix(y_test,prediction_Stack))

# evaluate Model Score
print(classification_report(y_test, prediction_Stack, digits=2))
#determining false positive rate and True positive rate, threshold
fpr, tpr, threshold = metrics.roc_curve(y_test, prediction_Stack)
roc_auc_stack = metrics.auc(fpr, tpr)
# print AUC
print("AUC : % 1.4f" %(roc_auc_stack)) 
#plotting ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_stack)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#creating model of Random Forest
RandomForest = RandomForestClassifier(n_estimators = 100,criterion='entropy',max_features=10)
RandomForest = RandomForest.fit(scaler_x_train, y_train)

# predict the response
RandomForest_pred = RandomForest.predict(scaler_x_test)


# Let's measure the accuracy of this model's prediction
print("confusion_matrix")
print(confusion_matrix(y_test,RandomForest_pred))

# evaluate Model Score
print(classification_report(y_test, RandomForest_pred, digits=2))
#determining false positive rate and True positive rate, threshold
fpr, tpr, threshold = metrics.roc_curve(y_test, RandomForest_pred)
roc_auc_rf = metrics.auc(fpr, tpr)
# print AUC
print("AUC : % 1.4f" %(roc_auc_rf)) 
#plotting ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_rf)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# Lets check features importance
feature_imp = pd.Series(RandomForest.feature_importances_,index=X_train.columns).sort_values(ascending=False)
feature_imp
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
#create and fit the model
AdBs = AdaBoostClassifier( n_estimators= 50)
AdBs  = AdBs.fit(scaler_x_train, y_train)

# predict the response
AdBs_pred = AdBs.predict(scaler_x_test)

# Let's measure the accuracy of this model's prediction
print("confusion_matrix")
print(confusion_matrix(y_test,AdBs_pred))

# evaluate Model Score
print(classification_report(y_test, AdBs_pred, digits=2))
#determining false positive rate and True positive rate, threshold
fpr, tpr, threshold = metrics.roc_curve(y_test, AdBs_pred)
roc_auc_ada = metrics.auc(fpr, tpr)
# print AUC
print("AUC : % 1.4f" %(roc_auc_ada)) 
#plotting ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_ada)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#Using K fold to check how my algorighm varies throughout my data if we split it in 10 equal bins
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('K-NN', KNeighborsClassifier(n_neighbors = 29, weights = 'uniform', metric='euclidean')))
models.append(('SVM', SVC(gamma=0.05, C=3)))
models.append(('Stacking', StackingClassifier(estimators=level0, final_estimator=level1, cv=5)))
models.append(('Random Forest', RandomForestClassifier(n_estimators = 100,criterion='entropy',max_features=10)))
models.append(('Adaptive Boosting', AdaBoostClassifier( n_estimators= 50)))

# evaluate each model
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=101)
	cv_results = model_selection.cross_val_score(model, scaler_x_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	print("Name = %s , Mean Accuracy = %f, SD Accuracy = %f" % (name, cv_results.mean(), cv_results.std()))
# boxplot algorithm comparison
fig = plt.figure(figsize=(12,4))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(names)