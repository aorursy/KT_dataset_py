import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

sb.set_style('darkgrid')

sb.set_palette('dark')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
data
data.isnull().sum()
data.dtypes
data.drop('time',1,inplace=True)
data.describe()
sb.pairplot(data,hue='DEATH_EVENT',aspect=0.8)
plt.figure(figsize=(20,6))

sb.heatmap(data.corr(),annot=True)
fig,ax = plt.subplots(1,2,figsize=(20,9))

ax[0].hist(data['age'],color = 'darkorange',label = 'patients',edgecolor='black')

ax[0].set_xlabel('Age')

ax[0].set_ylabel('Number of Patients')

ax[0].set_yticks([5,10,15,20,25,30,35,40,45,50,55,60])

ax[0].legend()

ax[0].set_title('Age Distribution')

ax[1].hist(x = [data[data['DEATH_EVENT']==1]['age'],data[data['DEATH_EVENT']==0]['age']],color = ['blue','darkorange'],stacked=True,edgecolor='black',label=['Dead','Survived'])

ax[1].set_xlabel('Age')

ax[1].set_ylabel('Number of patients')

ax[1].set_yticks([5,10,15,20,25,30,35,40,45,50,55,60])

ax[1].set_title('Dead and Survived patients by age')

ax[1].legend()
plt.figure(figsize = (20,8))

sb.regplot(data['DEATH_EVENT'],data['age'],color='darkorange')
fig,ax = plt.subplots(1,2,figsize=(20,9))

ax[0].hist(data['anaemia'],color='white',label = 'patients',edgecolor='black')

ax[0].set_xticks([0,1])

ax[0].set_yticks([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170])

ax[0].set_xticklabels(['Absent','Present'])

ax[0].set_xlabel('Anaemia')

ax[0].set_ylabel('Number of Patients')

ax[0].legend()

ax[0].set_title('anaemia presence in patients')

ax[1].hist(x = [data[data['DEATH_EVENT']==1]['anaemia'],data[data['DEATH_EVENT']==0]['anaemia']],stacked=True,color = ['blue','white'],edgecolor='black',label=['Dead','Survived'])

ax[1].set_xticks([0,1])

ax[1].set_xticklabels(['Absent','Present'])

ax[1].set_yticks([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170])

ax[1].set_xlabel('Anaemia')

ax[1].set_ylabel('Number of patients')

ax[1].set_title('dead and alive patients by presence of anaemia')

ax[1].legend()

plt.tight_layout()
plt.figure(figsize = (20,6))

sb.regplot(data['DEATH_EVENT'],data['anaemia'],color='red')

plt.title('DEATH due to anaemia')
fig,ax = plt.subplots(1,2,figsize=(20,9))

ax[0].hist(data['creatinine_phosphokinase'],color='red',label = 'patients',edgecolor='black')

ax[0].set_yticks([10,30,50,70,90,110,130,150,170,190,210,230,250])

ax[0].set_xticks([500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500])

ax[0].set_xlabel('creatinine phosphokinase')

ax[0].set_ylabel('Number of Patients')

ax[0].legend()

ax[0].set_title('creatinine phosphokinase levels')

ax[1].hist(x = [data[data['DEATH_EVENT']==1]['creatinine_phosphokinase'],data[data['DEATH_EVENT']==0]['creatinine_phosphokinase']],stacked=True,color = ['pink','red'],edgecolor='black',label=['Dead','Survived'])

ax[1].set_yticks([10,30,50,70,90,110,130,150,170,190,210,230,250])

ax[1].set_xticks([500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500])

ax[1].set_xlabel('creatinine phosphokinase')

ax[1].set_ylabel('Number of patients')

ax[1].set_title('creatinine phosphokinase levels and death event')

ax[1].legend()
plt.figure(figsize=(15,6))

sb.regplot(data['creatinine_phosphokinase'],data['DEATH_EVENT'],color='magenta')

plt.title('creatinine_phosphokinase levels and chance of death')
plt.figure(figsize=(15,6))

sb.regplot(data['creatinine_phosphokinase'],data['age'],color='magenta')

plt.title('creatinine_phosphokinase levels and age')
plt.figure(figsize=(15,6))

sb.regplot(data['creatinine_phosphokinase'],data['anaemia'],color='magenta')

plt.title('creatinine_phosphokinase levels and anaemia')
fig,ax = plt.subplots(1,2,figsize=(20,9))

ax[0].hist(data['diabetes'],color = 'grey',label = 'patients',edgecolor='black')

ax[0].set_xlabel('Diabetes')

ax[0].set_ylabel('Number of Patients')

ax[0].set_yticks([20,40,60,80,100,120,140,160,180,200])

ax[0].set_xticks([0,1])

ax[0].set_xticklabels(['Absent','Present'])

ax[0].legend()

ax[0].set_title('Diabetic Patients')

ax[1].hist(x = [data[data['DEATH_EVENT']==1]['diabetes'],data[data['DEATH_EVENT']==0]['diabetes']],color = ['green','grey'],stacked=True,edgecolor='black',label=['Dead','Survived'])

ax[1].set_xlabel('Diabetes')

ax[1].set_ylabel('Number of patients')

ax[1].set_yticks([20,40,60,60,80,100,120,140,160,180,200])

ax[1].set_xticks([0,1])

ax[1].set_xticklabels(['Absent','Present'])

ax[1].set_title('Dead and Survived patients by diabetic patients')

ax[1].legend()
plt.figure(figsize = (15,7))

sb.regplot(data['DEATH_EVENT'],data['diabetes'],color='green')
fig,ax = plt.subplots(1,2,figsize=(20,9))

ax[0].hist(data['ejection_fraction'],color = 'tomato',label = 'patients',edgecolor='black')

ax[0].set_xlabel('Ejection Fraction')

ax[0].set_ylabel('Number of patients')

ax[0].set_yticks([20,40,60,80,100,120,125])

ax[0].legend()

ax[0].set_title('Ejection Fraction')

ax[1].hist(x = [data[data['DEATH_EVENT']==1]['ejection_fraction'],data[data['DEATH_EVENT']==0]['ejection_fraction']],color = ['pink','tomato'],stacked=True,edgecolor='black',label=['Dead','Survived'])

ax[1].set_xlabel('Ejection Fraction')

ax[1].set_ylabel('Number of patients')

ax[1].set_yticks([20,40,60,80,100,120,125])

ax[1].set_title('Dead and Survived patients by Ejection Fraction')

ax[1].legend()
plt.figure(figsize = (20,7))

sb.regplot(data['DEATH_EVENT'],data['ejection_fraction'],color='tomato')

plt.title('Ejection Fraction vs Death event')
fig,ax = plt.subplots(1,2,figsize=(20,9))

ax[0].hist(data['high_blood_pressure'],color = 'yellow',label = 'patients',edgecolor='black')

ax[0].set_xlabel('High BP')

ax[0].set_ylabel('Number of Patients')

ax[0].set_yticks([20,40,60,80,100,120,140,160,180,200])

ax[0].set_xticks([0,1])

ax[0].set_xticklabels(['Absent','Present'])

ax[0].legend()

ax[0].set_title('High BP Patients')

ax[1].hist(x = [data[data['DEATH_EVENT']==1]['high_blood_pressure'],data[data['DEATH_EVENT']==0]['high_blood_pressure']],color = ['green','yellow'],stacked=True,edgecolor='black',label=['Dead','Survived'])

ax[1].set_xlabel('High BP')

ax[1].set_ylabel('Number of patients')

ax[1].set_yticks([20,40,60,60,80,100,120,140,160,180,200])

ax[1].set_xticks([0,1])

ax[1].set_xticklabels(['Absent','Present'])

ax[1].set_title('Dead and Survived patients By High BP')

ax[1].legend()

plt.figure(figsize = (20,7))

sb.regplot(data['high_blood_pressure'],data['DEATH_EVENT'],color='yellow')
fig,ax = plt.subplots(1,2,figsize=(25,9))

ax[0].hist(data['platelets'],color = 'red',edgecolor='black')

ax[0].set_xlabel('Platelet Count')

ax[0].set_ylabel('Number of patients')

ax[0].set_xticks([100000,200000,300000,400000,500000,600000,700000,800000])

ax[0].set_yticks([20,40,60,80,100,120,140])

ax[0].legend()

ax[0].set_title('Platelet Count')

ax[1].hist(x = [data[data['DEATH_EVENT']==1]['platelets'],data[data['DEATH_EVENT']==0]['platelets']],color = ['white','red'],stacked=True,edgecolor='black',label=['Dead','Survived'])

ax[1].set_xlabel('Platelet Count')

ax[1].set_ylabel('Number of patients')

ax[1].set_yticks([20,40,60,80,100,120,140])

ax[1].set_xticks([100000,200000,300000,400000,500000,600000,700000,800000])

ax[1].set_title('Dead and Survived patients and platelet count')

ax[1].legend()
plt.figure(figsize = (20,7))

sb.regplot(data['platelets'],data['DEATH_EVENT'],color='red')
plt.figure(figsize = (20,7))

sb.regplot(data['platelets'],data['age'],color='red')
fig,ax = plt.subplots(1,2,figsize=(25,9))

ax[0].hist(data['serum_creatinine'],color = 'darkgreen',edgecolor='black')

ax[0].set_xlabel('serum creatinine level')

ax[0].set_ylabel('Number of patients')

ax[0].set_yticks([5,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,220])

ax[0].set_xticks([1,2,3,4,5,6,7,8,9])

ax[0].legend()

ax[0].set_title('Serum Creatinine level')

ax[1].hist(x = [data[data['DEATH_EVENT']==1]['serum_creatinine'],data[data['DEATH_EVENT']==0]['serum_creatinine']],color = ['white','darkgreen'],stacked=True,edgecolor='black',label=['Dead','Survived'])

ax[1].set_xlabel('Serum Creatinine level')

ax[1].set_ylabel('Number of patients')

ax[1].set_yticks([5,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,220])

ax[1].set_xticks([1,2,3,4,5,6,7,8,9])

ax[1].set_title('Dead and Survived patients and Serum Creatinine level')

ax[1].legend()
plt.figure(figsize = (20,7))

sb.regplot(data['serum_creatinine'],data['DEATH_EVENT'],color='red')
fig,ax = plt.subplots(1,2,figsize=(25,9))

ax[0].hist(data['serum_sodium'],color = 'pink',edgecolor='black')

ax[0].set_xlabel('sodium level')

ax[0].set_ylabel('Number of patients')

ax[0].set_yticks([5,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,220])



ax[0].legend()

ax[0].set_title('Sodium level')

ax[1].hist(x = [data[data['DEATH_EVENT']==1]['serum_sodium'],data[data['DEATH_EVENT']==0]['serum_sodium']],color = ['magenta','pink'],stacked=True,edgecolor='black',label=['Dead','Survived'])

ax[1].set_xlabel('Sodium level')

ax[1].set_ylabel('Number of patients')

ax[1].set_yticks([5,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,220])



ax[1].set_title('Dead and Survived patients and Sodim level')

ax[1].legend()

plt.figure(figsize = (20,7))

sb.regplot(data['serum_sodium'],data['DEATH_EVENT'],color='red')

plt.title("Sodium Level vs Death event")
plt.figure(figsize = (20,7))

sb.regplot(data['serum_sodium'],data['ejection_fraction'],color='red')

plt.title("Sodium Level vs ejection Fraction")
plt.figure(figsize = (20,7))

sb.regplot(data['serum_sodium'],data['serum_creatinine'],color='red')

plt.title("Sodium bs creatinine")
fig,ax = plt.subplots(1,2,figsize=(20,9))

ax[0].hist(data['sex'],color = 'blue',label = 'patients',edgecolor='black')

ax[0].set_xlabel('Gender')

ax[0].set_ylabel('Number of Patients')

ax[0].set_yticks([20,40,60,80,100,120,140,160,180,200,220])

ax[0].set_xticks([0,1])

ax[0].set_xticklabels(['Female','Male'])

ax[0].legend()

ax[0].set_title('Gender Distribution')

ax[1].hist(x = [data[data['DEATH_EVENT']==1]['sex'],data[data['DEATH_EVENT']==0]['sex']],color = ['blue','black'],stacked=True,edgecolor='white',label=['Dead','Survived'])

ax[1].set_xlabel('Gender')

ax[1].set_ylabel('Number of patients')

ax[1].set_yticks([20,40,60,60,80,100,120,140,160,180,200,220])

ax[1].set_xticks([0,1])

ax[1].set_xticklabels(['Female','Male'])

ax[1].set_title('Dead and Survived patients By Gender')

ax[1].legend()

plt.figure(figsize = (20,7))

sb.regplot(data['sex'],data['DEATH_EVENT'],color='red')

plt.title("Gender vs Death event")
fig,ax = plt.subplots(1,2,figsize=(20,9))

ax[0].hist(data['smoking'],color = 'limegreen',label = 'patients',edgecolor='green')

ax[0].set_xlabel('Smoking')

ax[0].set_ylabel('Number of Patients')

ax[0].set_yticks([20,40,60,80,100,120,140,160,180,200,220])

ax[0].set_xticks([0,1])

ax[0].set_xticklabels(['No','Yes'])

ax[0].legend()

ax[0].set_title('Smoking Distribution')

ax[1].hist(x = [data[data['DEATH_EVENT']==1]['smoking'],data[data['DEATH_EVENT']==0]['smoking']],color = ['red','green'],stacked=True,edgecolor='black',label=['Dead','Survived'])

ax[1].set_xlabel('Smoking')

ax[1].set_ylabel('Number of patients')

ax[1].set_yticks([20,40,60,60,80,100,120,140,160,180,200,220])

ax[1].set_xticks([0,1])

ax[1].set_xticklabels(['No','Yes'])

ax[1].set_title('Dead and Survived patients By whether they smoke')

ax[1].legend()

plt.figure(figsize = (20,7))

sb.regplot(data['smoking'],data['DEATH_EVENT'],color='red')

plt.title("Smoking vs Death event")
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
y = data['DEATH_EVENT']

data.drop('DEATH_EVENT',1,inplace=True)

X = data



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf1 = LogisticRegression(C=.1)

clf1.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf1,X_train,y_train,cv=4)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf1,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf1,X_test,y_test,cv=4,scoring='accuracy').mean())
clf2 = SGDClassifier()

clf2.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf2,X_train,y_train,cv=4)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf2,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf2,X_test,y_test,cv=4,scoring='accuracy').mean())
param_grid = {'n_neighbors':np.arange(1,6)}

grid_knn = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)

grid_knn.fit(X,y)
grid_knn.best_params_
clf3 = grid_knn.best_estimator_

clf3.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf3,X_train,y_train,cv=4)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf3,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf3,X_test,y_test,cv=4,scoring='accuracy').mean())
param_grid = {'max_depth':np.arange(1,5),'min_samples_leaf':np.arange(1,4)}

grid_tree = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)

grid_tree.fit(X,y)
grid_tree.best_params_
clf4 = grid_tree.best_estimator_

clf4.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf4,X_train,y_train,cv=4)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf4,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf4,X_test,y_test,cv=4,scoring='accuracy').mean())
clf5 = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=3, max_features='auto',

                       max_leaf_nodes=2, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=2, min_samples_split=15,

                       min_weight_fraction_leaf=0.0, n_estimators=10,

                       n_jobs=-1, random_state=1, verbose=0,

                       warm_start=False)

clf5.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf5,X_train,y_train,cv=3)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf5,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf5,X_test,y_test,cv=4,scoring='accuracy').mean())
clf6 = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 3,min_samples_leaf=3),n_estimators = 1500)

clf6.fit(X_train,y_train)
y_train_pred = cross_val_predict(clf6,X_train,y_train,cv=3)

print("Confusion Matrix: \n",confusion_matrix(y_train,y_train_pred))

print('Precision Score:',precision_score(y_train,y_train_pred))

print("Recall Score:",recall_score(y_train,y_train_pred))

print("Accuracy Score:",accuracy_score(y_train,y_train_pred))

print("Cross Val Score Insample",cross_val_score(clf6,X_train,y_train,cv=4,scoring='accuracy').mean())

print("Cross Val Score Outsample",cross_val_score(clf6,X_test,y_test,cv=4,scoring='accuracy').mean())
classifier = DecisionTreeClassifier(max_depth=3,min_samples_leaf=3)

classifier.fit(X,y)