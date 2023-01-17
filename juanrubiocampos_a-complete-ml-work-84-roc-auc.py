# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
diabetes= pd.read_csv("../input/diabetes.csv")
print (diabetes.shape)
print ("--"*30)
print (diabetes.info())

diabetes.head()
diabetes.describe()
# class distribution
print(" Outcome distribution")
print(diabetes.groupby('Outcome').size())
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
#1) Replace 0 values to NaN values. Then sum the null values in each of those features,
#to know how many null values we have.
diabetes_copy=diabetes.copy(deep=True) ## We will need later the diabetes dataset with the 0s.

diabetes[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]]=diabetes[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.NaN)
print (diabetes.isnull().sum())
diabetes.describe()
# We replace the NaN values with the mean or median.
# Glucose and BloodPressure dont have much outliers, and we need little data to fill. The mean will be enough.
# The others, has a huge disparity between some samples, and we need a lot of data. So the median is best.
diabetes["Glucose"].fillna(diabetes["Glucose"].mean(),inplace=True)
diabetes["BloodPressure"].fillna(diabetes["BloodPressure"].mean(),inplace=True)
diabetes["SkinThickness"].fillna(diabetes["SkinThickness"].median(),inplace=True)
diabetes["Insulin"].fillna(diabetes["Insulin"].median(),inplace=True)
diabetes["BMI"].fillna(diabetes["BMI"].median(),inplace=True)

print (diabetes.isnull().sum())
print ('--'*40)
diabetes.info()
print ('--'*40)
diabetes.head()
diabetes.describe()
plt.figure(figsize=(10,6))
sns.distplot(diabetes['Pregnancies'],kde=False,bins=50)
plt.title('Pregnancies per Person on Pima People')
plt.ylabel('Number of People')
plt.show()
print('Average amount of children had by a Pima woman: ' + str(diabetes['Pregnancies'].mean()))

plt.figure(figsize=(10,6))
sns.distplot(diabetes['Glucose'],kde=False,bins=50)
plt.title('Glucose per Person on Pima People')
plt.ylabel('Number of People')
plt.show()
plt.figure(figsize=(10,6))
sns.distplot(diabetes['BloodPressure'],kde=False,bins=50)
plt.title('Blood Pressure of Pima Indian People')
plt.ylabel('Number of People')
plt.show()
plt.figure(figsize=(10,6))
sns.distplot(diabetes['SkinThickness'],kde=False,bins=50)
plt.title('Skin Thickness of Pima Indian People')
plt.ylabel('Number of People')
plt.show()
plt.figure(figsize=(10,6))
sns.distplot(diabetes['Insulin'],kde=False,bins=50)
plt.title('Insulin of Pima Indian People')
plt.ylabel('Number of People')
plt.show()
plt.figure(figsize=(10,6))
sns.distplot(diabetes['BMI'],kde=False,bins=50)
plt.title('BMI of Pima Indian People')
plt.ylabel('Number of People')
plt.show()
print('Average BMI of a Pima Person: ' + str(diabetes['BMI'].mean()))
plt.figure(figsize=(10,6))
sns.distplot(diabetes['DiabetesPedigreeFunction'],kde=False,bins=50)
plt.title('Diabetes Pedigree Function of Pima Indian People')
plt.ylabel('Number of People')
plt.show()
plt.figure(figsize=(10,6))
sns.distplot(diabetes['Age'],kde=False,bins=50)
plt.title('Age of Pima Indian People')
plt.ylabel('Number of People')
plt.show()
plt.figure(figsize=(8,6))
sns.countplot(x='Outcome',data=diabetes)
plt.title('Positive Outcome to Diabetes in Dataset')
plt.ylabel('Number of People')
plt.show()
print('Ratio of Population with Diabetes: ' + str(len(diabetes[diabetes['Outcome']==1])/len(diabetes)))
plt.figure(figsize=(10,6))
sns.heatmap(diabetes.corr(),cmap='YlGn',annot=True)
plt.show()
g = sns.FacetGrid(diabetes, col="Outcome",size=5)
g = g.map(plt.hist, "Glucose",bins=30)
print('Average number of glucose for positive outcomes: ' + str(diabetes[diabetes['Outcome']==1]['Glucose'].mean()))
print('Average number of glucose for negative outcomes: ' + str(diabetes[diabetes['Outcome']==0]['Glucose'].mean()))
g = sns.FacetGrid(diabetes, col="Outcome",size=5)
g = g.map(plt.hist, "BMI",bins=30)
print('Average Body Mass Index of a Pima woman without diabetes: ' + str(diabetes[diabetes['Outcome']==0]['BMI'].mean()))
print('Average Body Mass Index of a Pima woman with diabetes: ' + str(diabetes[diabetes['Outcome']==1]['BMI'].mean()))
plt.figure(figsize=(10,6))
sns.barplot(data=diabetes,x='Outcome',y='Pregnancies')
plt.title('Pregnancies Among Diabetes Outcomes.')
plt.show()
plt.figure(figsize=(10,6))
sns.countplot(x='Pregnancies',data=diabetes,hue='Outcome')
plt.title('Diabetes Outcome to Pregnancies')
plt.show()
print('Average number of pregnancies for positive outcomes: ' + str(diabetes[diabetes['Outcome']==1]['Pregnancies'].mean()))
print('Average number of pregnancies for negative outcomes: ' + str(diabetes[diabetes['Outcome']==0]['Pregnancies'].mean()))
plt.figure(figsize=(13,6))
sns.countplot(x='Age',data=diabetes,hue='Outcome')
plt.title('Diabetes Outcome to Age')
plt.show()
print('Average number of age for positive outcomes: ' + str(diabetes[diabetes['Outcome']==1]['Age'].mean()))
print('Average number of age for negative outcomes: ' + str(diabetes[diabetes['Outcome']==0]['Age'].mean()))
plt.figure(figsize=(13,6))
sns.countplot(x='SkinThickness',data=diabetes,hue='Outcome')
plt.title('Diabetes Outcome to SkinThickness')
plt.show()
print('Average number of skin thickness for positive outcomes: ' + str(diabetes[diabetes['Outcome']==1]['SkinThickness'].mean()))
print('Average number of skin thickness for negative outcomes: ' + str(diabetes[diabetes['Outcome']==0]['SkinThickness'].mean()))
# Diabetes has a lot of values, so to plot it I need to make a few changes. The most important one is to
# divide it in quartiles.
diabetes_copy2= diabetes.copy(deep=True)
diabetes_copy2["InsulinBins"]=pd.qcut(diabetes["Insulin"],4)
#Now we can plot
plt.figure(figsize=(13,6))
sns.countplot(x='InsulinBins',data=diabetes_copy2,hue='Outcome')
plt.title('Diabetes Outcome to Insulin')
plt.show()
print('Average number of Insulin for positive outcomes: ' + str(diabetes[diabetes['Outcome']==1]['Insulin'].mean()))
print('Average number of Insulin for negative outcomes: ' + str(diabetes[diabetes['Outcome']==0]['Insulin'].mean()))
diabetes_copy2["DiabetesPedigreeBins"]=pd.qcut(diabetes["DiabetesPedigreeFunction"],4)
# Same reason as in insulin
plt.figure(figsize=(13,6))
sns.countplot(x='DiabetesPedigreeBins',data=diabetes_copy2,hue='Outcome')
plt.title('Diabetes Outcome to DiabetesPedigreeFunction')
plt.show()
print('Average number of Diabetes Pedigree Function for positive outcomes: ' + str(diabetes[diabetes['Outcome']==1]['DiabetesPedigreeFunction'].mean()))
print('Average number of Diabetes Pedigree Function for negative outcomes: ' + str(diabetes[diabetes['Outcome']==0]['DiabetesPedigreeFunction'].mean()))
diabetes_copy2["GlucoseBins"]=pd.qcut(diabetes["Glucose"],4)
plt.figure(figsize=(13,6))
sns.countplot(x='GlucoseBins',data=diabetes_copy2,hue='DiabetesPedigreeBins')
plt.title('Glucose and Diabetes Pedigree relationship.')
plt.show()
plt.figure(figsize=(13,6))
sns.countplot(x='BloodPressure',data=diabetes,hue='Outcome')
plt.title('Diabetes Outcome to BloodPressure')
plt.show()
print('Average number of Blood Presure for positive outcomes: ' + str(diabetes[diabetes['Outcome']==1]['BloodPressure'].mean()))
print('Average number of Blood Pressure for negative outcomes: ' + str(diabetes[diabetes['Outcome']==0]['BloodPressure'].mean()))
diabetes_copy2["BloodPressureBins"]=pd.qcut(diabetes["BloodPressure"],4)
diabetes_copy2["BMIBins"]=pd.qcut(diabetes["BMI"],4)

plt.figure(figsize=(13,6))
sns.countplot(x='BloodPressureBins',data=diabetes_copy2,hue='BMIBins')
plt.title('BMI relation with BloodPressure')
plt.show()
diabetes.hist(figsize=(8,8))
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE  
from imblearn.pipeline import Pipeline as Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import (GridSearchCV,StratifiedKFold)
X=diabetes_copy.drop(["Outcome"], axis=1)
y=diabetes_copy["Outcome"]
scoring = 'roc_auc'
seed=7
models = [] # Here I will append all the algorithms that I will use. Each one will run in all the created datasets.
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC()))
models.append(('AdaBoost', AdaBoostClassifier()))

print("evaluation metric: " + scoring)    
results=[]
names=[]
for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model,X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        
        
        print ("Algorithm :",name)
        print (" Baseline CV mean: ", cv_results.mean())
        print ("--"*30)

train_X,test_X,train_y,test_y = train_test_split (X,y,test_size=0.2,random_state=3)

model1= LogisticRegression()
fit1 =model1.fit(train_X,train_y)
prediction1= model1.predict(test_X)
confusion= metrics.confusion_matrix(test_y, prediction1)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print ("Baseline model accuracy: ", metrics.accuracy_score(test_y,prediction1))
print ("--"*30)
print ("Baseline matrix confusion: ", "\n",metrics.confusion_matrix(test_y,prediction1))
print ("--"*30)
print ("Baseline sensitivity: ", TP / float(FN + TP))
print ("--"*30)
print ("Baseline model specificity: ", TN / (TN + FP))
print ("--"*30)
print ("Baseline roc auc score: ", "\n", metrics.roc_auc_score(test_y,prediction1))
X=diabetes_copy.drop(["Outcome"], axis=1)
y=diabetes_copy["Outcome"]

train_X,test_X,train_y,test_y = train_test_split (X,y,test_size=0.2,random_state=3)

model2= LinearDiscriminantAnalysis()
fit2 =model2.fit(train_X,train_y)
prediction2= model2.predict(test_X)
confusion= metrics.confusion_matrix(test_y, prediction2)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print ("Baseline model accuracy: ", metrics.accuracy_score(test_y,prediction2))
print ("--"*30)
print ("Baseline matrix confusion: ", "\n",metrics.confusion_matrix(test_y,prediction2))
print ("--"*30)
print ("Baseline sensitivity: ", TP / float(FN + TP))
print ("--"*30)
print ("Baseline model specificity: ", TN / (TN + FP))
print ("--"*30)
print ("Baseline roc auc score: ", "\n", metrics.roc_auc_score(test_y,prediction2))
X=diabetes.drop(["Outcome"], axis=1)
y=diabetes["Outcome"]
scoring = 'roc_auc'
seed=7
models = [] # Here I will append all the algorithms that I will use. Each one will run in all the created datasets.
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC()))
models.append(('AdaBoost', AdaBoostClassifier()))

print("evaluation metric: " + scoring)    
results=[]
names=[]
for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model,X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        
        
        print ("Algorithm :",name)
        print (" Data clean CV mean: ", cv_results.mean())
        print ("--"*30)

train_X,test_X,train_y,test_y = train_test_split (X,y,test_size=0.2,random_state=3)

model3=LogisticRegression()
fit3 =model3.fit(train_X,train_y)
prediction3= model3.predict(test_X)
confusion= metrics.confusion_matrix(test_y, prediction3)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print ("Data clean model roc auc score: ",metrics.roc_auc_score(test_y,prediction3))
print ("__"*30)
print ("Data clean model matrix confusion: ", "\n",metrics.confusion_matrix(test_y,prediction3))
print ("--"*30)
print ("Data clean model sensitivity: ", TP / float(FN + TP))
print ("--"*30)
print ("Data clean model specificity: ", TN / (TN + FP))
print ("__"*30)
print ("Data clean model accuracy: ", metrics.accuracy_score(test_y,prediction3))

train_X,test_X,train_y,test_y = train_test_split (X,y,test_size=0.2,random_state=3)

model4=LinearDiscriminantAnalysis()
fit4 =model4.fit(train_X,train_y)
prediction4= model4.predict(test_X)
confusion= metrics.confusion_matrix(test_y, prediction4)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print ("Data clean model roc auc score: ",metrics.roc_auc_score(test_y,prediction4))
print ("__"*30)
print ("Data clean model matrix confusion: ", "\n",metrics.confusion_matrix(test_y,prediction4))
print ("--"*30)
print ("Data clean model sensitivity: ", TP / float(FN + TP))
print ("--"*30)
print ("Data clean model specificity: ", TN / (TN + FP))
print ("__"*30)
print ("Data clean model accuracy: ", metrics.accuracy_score(test_y,prediction4))
from sklearn.preprocessing import StandardScaler
X=diabetes.drop(["Outcome"],axis=1)
print (X.info())
columnas=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X=pd.DataFrame(X_scaled, columns=[columnas])

X.head()
scoring = 'roc_auc'
seed=7
models = [] # Here I will append all the algorithms that I will use. Each one will run in all the created datasets.
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC()))
models.append(('AdaBoost', AdaBoostClassifier()))

print("evaluation metric: " + scoring)    
results=[]
names=[]
for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model,X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        
        
        print ("Algorithm :",name)
        print (" Data clean & scaled CV mean: ", cv_results.mean())
        print ("--"*30)
train_X,test_X,train_y,test_y = train_test_split (X,y,test_size=0.2,random_state=3)

model5= LogisticRegression()
fit5 =model5.fit(train_X,train_y)
prediction5= model5.predict(test_X)
confusion= metrics.confusion_matrix(test_y, prediction5)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print ("Data clean & scaled roc auc score: ", "\n", metrics.roc_auc_score(test_y,prediction5))
print ("--"*30)
print ("Data clean & scaled matrix confusion: ", "\n",metrics.confusion_matrix(test_y,prediction5))
print ("--"*30)
print ("Data clean & scaled model sensitivity: ", TP / float(FN + TP))
print ("--"*30)
print ("Data clean & scaled model specificity: ", TN / (TN + FP))
print ("__"*30)
print ("Data clean & scaled model accuracy: ", metrics.accuracy_score(test_y,prediction5))

train_X,test_X,train_y,test_y = train_test_split (X,y,test_size=0.2,random_state=3)

model= LinearDiscriminantAnalysis()
fit6 =model.fit(train_X,train_y)
prediction6= model.predict(test_X)
confusion= metrics.confusion_matrix(test_y, prediction6)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print ("Data clean & scaled roc auc score: ", "\n", metrics.roc_auc_score(test_y,prediction6))
print ("--"*30)
print ("Data clean & scaled matrix confusion: ", "\n",metrics.confusion_matrix(test_y,prediction6))
print ("--"*30)
print ("Data clean & scaled model sensitivity: ", TP / float(FN + TP))
print ("--"*30)
print ("Data clean & scaled model specificity: ", TN / (TN + FP))
print ("__"*30)
print ("Data clean & scaled model accuracy: ", metrics.accuracy_score(test_y,prediction6))

from imblearn.over_sampling import SMOTE  
from imblearn.pipeline import Pipeline as Pipeline

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV,StratifiedKFold)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=444, stratify=y)

pipe = Pipeline([
    ('oversample', SMOTE(random_state=444)),
    ('clf', LogisticRegression(random_state=444, n_jobs=-1))
    ])

skf = StratifiedKFold(n_splits=10)
param_grid = {'clf__C': [0.001,0.01,0.1,1,10,100],
              'clf__penalty': ['l1', 'l2']}
grid = GridSearchCV(pipe, param_grid, return_train_score=False,
                    n_jobs=-1, scoring="roc_auc", cv=skf)
logreg=grid.fit(X_train, y_train)
grid.score(X_test, y_test)
# View best hyperparameters
prediction7= grid.predict(X_test)
confusion= metrics.confusion_matrix(y_test, prediction7)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print ("SMOTE roc auc score: ", "\n", metrics.roc_auc_score(y_test,prediction7))
print ("--"*30)
print ("SMOTE matrix confusion: ", "\n",metrics.confusion_matrix(y_test,prediction7))
print ("--"*30)
print ("SMOTE sensitivity: ", TP / float(FN + TP))
print ("--"*30)
print ("SMOTE model specificity: ", TN / (TN + FP))
print ("__"*30)
print ("SMOTE model accuracy: ", metrics.accuracy_score(y_test,prediction7))
print ("--"*30)

print('Best Penalty:', grid.best_estimator_.get_params()['clf__penalty'])
print('Best C:', grid.best_estimator_.get_params()['clf__C'])



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=444, stratify=y)

pipe2 = Pipeline([
    ('oversample', SMOTE(random_state=444)),
    ('clf', LinearDiscriminantAnalysis())])

skf2 = StratifiedKFold(n_splits=10)
param_grid = {'clf__n_components': [1]}
grid = GridSearchCV(pipe2, param_grid, return_train_score=False,
                    n_jobs=-1, scoring="roc_auc", cv=skf2)
LDA=grid.fit(X_train, y_train)
grid.score(X_test, y_test)
# View best hyperparameters
prediction8= LDA.predict(X_test)
confusion2= metrics.confusion_matrix(y_test, prediction8)
TP = confusion2[1, 1]
TN = confusion2[0, 0]
FP = confusion2[0, 1]
FN = confusion2[1, 0]

print ("SMOTE roc auc score: ", "\n", metrics.roc_auc_score(y_test,prediction8))
print ("--"*30)
print ("SMOTE matrix confusion: ", "\n",metrics.confusion_matrix(y_test,prediction8))
print ("--"*30)
print ("SMOTE sensitivity: ", TP / float(FN + TP))
print ("--"*30)
print ("SMOTE model specificity: ", TN / (TN + FP))
print ("__"*30)
print ("SMOTE model accuracy: ", metrics.accuracy_score(y_test,prediction8))
print ("--"*30)

print('Best Number of components:', grid.best_estimator_.get_params()['clf__n_components'])

logreg.predict(X_test)[0:10]
logreg.predict_proba(X_test)[0:10]
logreg.predict_proba(X_test)[0:10, 1]
y_pred_prob = logreg.predict_proba(X_test)[:, 1] # Here I saved all the samples, not just 10.
plt.hist(y_pred_prob, bins=8)

# x-axis limit from 0 to 1
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')
from sklearn.preprocessing import binarize
y_pred_prob=y_pred_prob.reshape(1,-1)
# it will return 1 for all values above 0.3 and 0 otherwise
# results are 2D so we slice out the first column
y_pred_class = binarize(y_pred_prob, 0.3)[0]
y_pred_prob[0:10]
y_pred_class[0:10]
y_pred_prob = logreg.predict_proba(X_test)[:, 1] # remember that equation
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
evaluate_threshold(0.5) # For example.
print(metrics.roc_auc_score(y_test, y_pred_prob))
from sklearn.cross_validation import cross_val_score
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()