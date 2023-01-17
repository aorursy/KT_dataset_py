#Importing Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Defining Data file

pData = pd.read_csv("/kaggle/input/parkinsons-data-set/parkinsons.data")
pData.head()
pData.shape
pData.columns
pData.info()
# Dropping name, not required for analysis(nominal variables)

pData= pData.drop(['name'], axis=1)
pData.head()
pData.skew()
pData.describe().T
#Distribution of target column



pData['status'] = pData['status'].astype('float') 



pData['status'].value_counts()
#Distribution of MDVP:Fo(Hz) - Average vocal fundamental frequency, MDVP:Fhi(Hz) - Maximum vocal fundamental frequency, MDVP:Flo(Hz) - Minimum vocal fundamental frequency

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.distplot(pData['MDVP:Fo(Hz)'],ax=axes[0])

sns.distplot(pData['MDVP:Fhi(Hz)'],ax=axes[1])

sns.distplot(pData['MDVP:Flo(Hz)'],ax=axes[2])
sns.distplot( pData[pData.status == 0]['MDVP:Fo(Hz)'], color = 'r')

sns.distplot( pData[pData.status == 1]['MDVP:Fo(Hz)'], color = 'g')
plt.figure(figsize=(8,5))

sns.boxplot(x='status',y='MDVP:Fo(Hz)',data=pData)
sns.distplot( pData[pData.status == 0]['MDVP:Fhi(Hz)'], color = 'r')

sns.distplot( pData[pData.status == 1]['MDVP:Fhi(Hz)'], color = 'g')
plt.figure(figsize=(8,5))

sns.boxplot(x='status',y='MDVP:Fhi(Hz)',data=pData)
sns.distplot( pData[pData.status == 0]['MDVP:Flo(Hz)'], color = 'r')

sns.distplot( pData[pData.status == 1]['MDVP:Flo(Hz)'], color = 'g')
plt.figure(figsize=(8,5))

sns.boxplot(x='status',y='MDVP:Flo(Hz)',data=pData)
#Analysis of measures of variation in fundamental frequency

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

sns.distplot(pData['MDVP:Jitter(%)'],bins=30,ax=axes[0,0])

sns.distplot(pData['MDVP:Jitter(Abs)'],bins=30,ax=axes[0,1])

sns.distplot(pData['MDVP:RAP'],bins=30,ax=axes[0,2])



sns.distplot(pData['MDVP:PPQ'],bins=30,ax=axes[1,0])

sns.distplot(pData['Jitter:DDP'],bins=30,ax=axes[1,1])

sns.distplot( pData[pData.status == 0]['MDVP:Jitter(%)'], color = 'r')

sns.distplot( pData[pData.status == 1]['MDVP:Jitter(%)'], color = 'g')
plt.figure(figsize=(8,5))

sns.boxplot(x='status',y='MDVP:Jitter(%)',data=pData)
sns.distplot( pData[pData.status == 0]['MDVP:Jitter(Abs)'], color = 'r')

sns.distplot( pData[pData.status == 1]['MDVP:Jitter(Abs)'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='MDVP:Jitter(Abs)',data=pData)
sns.distplot( pData[pData.status == 0]['MDVP:RAP'], color = 'r')

sns.distplot( pData[pData.status == 1]['MDVP:RAP'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='MDVP:RAP',data=pData)
sns.distplot( pData[pData.status == 0]['MDVP:PPQ'], color = 'r')

sns.distplot( pData[pData.status == 1]['MDVP:PPQ'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='MDVP:PPQ',data=pData)
sns.distplot( pData[pData.status == 0]['Jitter:DDP'], color = 'r')

sns.distplot( pData[pData.status == 1]['Jitter:DDP'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='Jitter:DDP',data=pData)
#Analysis of  variation in amplitude

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

sns.distplot(pData['MDVP:Shimmer'],ax=axes[0,0])

sns.distplot(pData['MDVP:Shimmer(dB)'],ax=axes[0,1])

sns.distplot(pData['Shimmer:APQ3'],ax=axes[0,2])



sns.distplot(pData['Shimmer:APQ5'],ax=axes[1,0])

sns.distplot(pData['MDVP:APQ'],ax=axes[1,1])

sns.distplot(pData['Shimmer:DDA'],ax=axes[1,2])
sns.distplot( pData[pData.status == 0]['MDVP:Shimmer'], color = 'r')

sns.distplot( pData[pData.status == 1]['MDVP:Shimmer'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='MDVP:Shimmer',data=pData)
sns.distplot( pData[pData.status == 0]['MDVP:Shimmer(dB)'], color = 'r')

sns.distplot( pData[pData.status == 1]['MDVP:Shimmer(dB)'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='MDVP:Shimmer(dB)',data=pData)
sns.distplot( pData[pData.status == 0]['Shimmer:APQ3'], color = 'r')

sns.distplot( pData[pData.status == 1]['Shimmer:APQ3'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='Shimmer:APQ3',data=pData)
sns.distplot( pData[pData.status == 0]['Shimmer:APQ5'], color = 'r')

sns.distplot( pData[pData.status == 1]['Shimmer:APQ5'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='Shimmer:APQ5',data=pData)
sns.distplot( pData[pData.status == 0]['MDVP:APQ'], color = 'r')

sns.distplot( pData[pData.status == 1]['MDVP:APQ'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='MDVP:APQ',data=pData)
sns.distplot( pData[pData.status == 0]['Shimmer:DDA'], color = 'r')

sns.distplot( pData[pData.status == 1]['Shimmer:DDA'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='Shimmer:DDA',data=pData)
#analysis for measures of ratio of noise to tonal components in the voice

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.distplot(pData['NHR'],ax=axes[0])

sns.distplot(pData['HNR'],ax=axes[1])

sns.distplot( pData[pData.status == 0]['NHR'], color = 'r')

sns.distplot( pData[pData.status == 1]['NHR'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='NHR',data=pData)
sns.distplot( pData[pData.status == 0]['HNR'], color = 'r')

sns.distplot( pData[pData.status == 1]['HNR'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='HNR',data=pData)
#analysis for wo nonlinear dynamical complexity measures

fig, axes = plt.subplots(1, 2, figsize=(20,8))

sns.distplot(pData['RPDE'],ax=axes[0])

sns.distplot(pData['D2'],ax=axes[1])
sns.distplot( pData[pData.status == 0]['RPDE'], color = 'r')

sns.distplot( pData[pData.status == 1]['RPDE'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='RPDE',data=pData)
sns.distplot( pData[pData.status == 0]['D2'], color = 'r')

sns.distplot( pData[pData.status == 1]['D2'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='D2',data=pData)
#Analysis of nonlinear measures of fundamental frequency variation 

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.distplot(pData['spread1'],ax=axes[0])

sns.distplot(pData['spread2'],ax=axes[1])

sns.distplot(pData['PPE'],ax=axes[2])
sns.distplot( pData[pData.status == 0]['spread1'], color = 'r')

sns.distplot( pData[pData.status == 1]['spread1'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='spread1',data=pData)
sns.distplot( pData[pData.status == 0]['spread2'], color = 'r')

sns.distplot( pData[pData.status == 1]['spread2'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='spread2',data=pData)
sns.distplot( pData[pData.status == 0]['PPE'], color = 'r')

sns.distplot( pData[pData.status == 1]['PPE'], color = 'g')
plt.figure(figsize=(4,5))

sns.boxplot(x='status',y='PPE',data=pData)
#Exploring the correlation

pData.corr()
corr = pData.corr()

fig, ax = plt.subplots(figsize = [23,10])

sns.heatmap(corr, annot = True, vmin = -1, vmax = 1, center = 0, cmap="coolwarm")
#Filtering highly related columns

corr_pos = corr.abs()

mask = (corr_pos < 0.8 ) 

fig, ax = plt.subplots(figsize = [23,10])

sns.heatmap(corr, annot = True, vmin = -1, vmax = 1, center = 0, mask = mask, cmap="coolwarm")
from scipy import stats

z=np.abs(stats.zscore(pData))

print(z)
print(np.where(z > 3))
z_pData =pData[(z < 3).all(axis=1)]

pData.shape,z_pData.shape
corr = z_pData.corr()

#Filtering highly related columns

corr_pos = corr.abs()

mask = (corr_pos < 0.8 ) 

fig, ax = plt.subplots(figsize = [23,10])

sns.heatmap(corr, annot = True, vmin = -1, vmax = 1, center = 0, mask = mask, cmap="coolwarm")
X = pData.drop(['status'], axis = 1)

y = pData['status']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=8)
#Scaling

from sklearn.preprocessing import StandardScaler



scaler=StandardScaler().fit(X_train)

scaler_x_train=scaler.transform(X_train)



scaler=StandardScaler().fit(X_test)

scaler_x_test=scaler.transform(X_test)
#For model metrics' calculation

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix



def clf_scores(clf,y_predicted):

    #Accuracy

    acc_train = clf.score(X_train, y_train)*100

    acc_test = clf.score(X_test, y_test)*100

    

    roc = roc_auc_score(y_test, y_predicted)*100

    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()

    cm = confusion_matrix(y_test, y_predicted)

    correct = tp + tn

    incorrect = fp + fn

        

    return acc_train, acc_test, roc, correct, incorrect, cm



def clf_table(clf,y_predicted):

    cr=classification_report(y_test,y_predicted, digits=2)

    return cr

#For capturing model's performance metrics

performance = pd.DataFrame(columns = ['Model', 'Accuracy', 'Recall', 'Precision', 'F1 Score', 'AUC'])

#Logistic regression



from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression()

clf_lr.fit(X_train, y_train)



Y_pred_lr = clf_lr.predict(X_test)

print(clf_scores(clf_lr, Y_pred_lr))
from sklearn import preprocessing

from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, classification_report,f1_score



cr1= clf_table(clf_lr, Y_pred_lr)

print(cr1)
from sklearn import metrics

#determining ROC-AUC using fpr, tpr, threshold

fpr, tpr, threshold = metrics.roc_curve(y_test, Y_pred_lr)

roc_auc_stack = metrics.auc(fpr, tpr)

print("AUC : % 1.4f" %(roc_auc_stack)) 
performance = performance.append({'Model':'Linear Regression','Accuracy':'86','Recall':'94','Precision':'90','F1 Score':'92','AUC':'75.98'},ignore_index = True)
#KNN



from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors = 3)

clf_knn.fit(X_train, y_train)



Y_pred_knn = clf_knn.predict(X_test)

print(clf_scores(clf_knn, Y_pred_knn))
cr2 =clf_table(y_test,Y_pred_knn)

print(cr2)
#determining ROC-AUC using fpr, tpr, threshold

fpr, tpr, threshold = metrics.roc_curve(y_test, Y_pred_knn)

roc_auc_stack = metrics.auc(fpr, tpr)

print("AUC : % 1.4f" %(roc_auc_stack)) 
performance = performance.append({'Model':'KNN','Accuracy':'85','Recall':'91','Precision':'90','F1 Score':'91','AUC':'74.91'},ignore_index = True)
#Naive Bayes



from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()

clf_gnb.fit(X_train, y_train)



Y_pred_gnb = clf_gnb.predict(X_test)

print(clf_scores(clf_gnb, Y_pred_gnb))
cr3=clf_table(y_test,Y_pred_gnb)

print(cr3)
#determining ROC-AUC using fpr, tpr, threshold

fpr, tpr, threshold = metrics.roc_curve(y_test, Y_pred_gnb)

roc_auc_stack = metrics.auc(fpr, tpr)

print("AUC : % 1.4f" %(roc_auc_stack)) 
performance = performance.append({'Model':'Naive Bayes','Accuracy':'71','Recall':'68','Precision':'94','F1 Score':'79','AUC':'75.71'},ignore_index = True)
#SVM



from sklearn.svm import SVC

clf_svc = SVC(gamma=0.05, C=3,random_state=0)

clf_svc.fit(X_train, y_train)



Y_pred_svc = clf_svc.predict(X_test)

print(clf_scores(clf_svc, Y_pred_svc))
cr4 =clf_table(y_test,Y_pred_svc)

print(cr4)
#determining ROC-AUC using fpr, tpr, threshold

fpr, tpr, threshold = metrics.roc_curve(y_test,Y_pred_svc)

roc_auc_stack = metrics.auc(fpr, tpr)

print("AUC : % 1.4f" %(roc_auc_stack)) 
performance = performance.append({'Model':'SVM','Accuracy':'85','Recall':'100','Precision':'84','F1 Score':'91','AUC':'62.5'},ignore_index = True)
print(performance)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import StackingClassifier



# defining level hetrogenious model

level0 = list()

level0.append(('lr', LogisticRegression()))

level0.append(('knn', KNeighborsClassifier(n_neighbors = 29, weights = 'uniform', metric='euclidean')))

level0.append(('cart', DecisionTreeClassifier()))

level0.append(('svm', SVC(gamma=0.05, C=3)))

level0.append(('bayes', GaussianNB()))



# define meta learner model

level1 = SVC(gamma=0.05, C=3)



# define the stacking ensemble with cross validation of 10

Stack_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10)

# predict the response

Stack_model.fit(scaler_x_train, y_train)

prediction_Stack = Stack_model.predict(scaler_x_test)
print(clf_scores(Stack_model, prediction_Stack))
cr5 = clf_table(Stack_model, prediction_Stack)

print(cr5)
#determining ROC-AUC using fpr, tpr, threshold

fpr, tpr, threshold = metrics.roc_curve(y_test, prediction_Stack)

roc_auc_stack = metrics.auc(fpr, tpr)

print("AUC : % 1.4f" %(roc_auc_stack)) 
performance = performance.append({'Model':'Stacking:SVM','Accuracy':'93','Recall':'98','Precision':'94','F1 Score':'96','AUC':'86.44'},ignore_index = True)
#Stacking using Logistic reg. 

# define meta learner model

level1 = LogisticRegression()



# define the stacking ensemble with cross validation of 10

Stack_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10)
# predict the response

Stack_model.fit(scaler_x_train, y_train)

prediction_Stack = Stack_model.predict(scaler_x_test)
print(clf_scores(Stack_model, prediction_Stack))
cr6 = clf_table(Stack_model, prediction_Stack)

print(cr6)
#determining ROC-AUC using fpr, tpr, threshold

fpr, tpr, threshold = metrics.roc_curve(y_test, prediction_Stack)

roc_auc_stack = metrics.auc(fpr, tpr)

print("AUC : % 1.4f" %(roc_auc_stack)) 
performance = performance.append({'Model':'Stacking:LG','Accuracy':'92','Recall':'98','Precision':'92','F1 Score':'95','AUC':'82.27'},ignore_index = True)
from sklearn.ensemble import RandomForestClassifier



rfcl = RandomForestClassifier(n_estimators = 50)

rfcl = rfcl.fit(X_train, y_train)

y_pred_rf = rfcl.predict(X_test)

print(clf_scores(rfcl, y_pred_rf))
cr7 = clf_table(rfcl, y_pred_rf)

print(cr7)
#determining ROC-AUC using fpr, tpr, threshold

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_rf)

roc_auc_stack = metrics.auc(fpr, tpr)

print("AUC : % 1.4f" %(roc_auc_stack))
performance = performance.append({'Model':'Random Forest','Accuracy':'93','Recall':'100','Precision':'92','F1 Score':'96','AUC':'83.3'},ignore_index = True)
print(performance)
performance.set_index("Model", inplace = True)
print(performance)
convert_dict = {'Accuracy': int, 'Recall': int,'Precision': int, 'F1 Score': int, 'AUC': float} 

summary = performance.astype(convert_dict) 
print(summary.dtypes)
plt.figure(figsize=(11,8))

sns.lineplot(data=summary)