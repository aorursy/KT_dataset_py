import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import os



print(os.listdir("../input"))
data = pd.read_csv('../input/heart.csv')



data.head()
data.isnull().sum()
data.drop([48,92,158,163,164,251,281], inplace=True)
data2 = data.copy()



data2.columns = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol', 'Fasting Blood Sugar', 'Rest ECG', 'Max Heart Rate Achieved',

       'Exercise Induced Angina', 'st Depression', 'st Slope', 'Number Major Vessels', 'Thalassemia', 'Heart Desease']



data2.replace({'Sex':{0:'Female',1:'Male'},

              'Chest Pain Type':{0:'Asymptomatic',1:'Atypical Angina',2:'Non-Anginal Pain',3:'Typical Angina'},

              'Fasting Blood Sugar':{1:'> 120 mg/dl',0:'< 120 mg/dl'},

              'Rest ECG':{0:'Left Ventricular Hypertrophy',1:'Normal',2:'ST-T wave abnormality'},

              'Exercise Induced Angina':{1:'Yes',0:'No'},

              'st Slope':{0:'Downsloping',1:'Flat',2:'Upsloping'},

              'Thalassemia':{1:'Fixed Defect',2:'Normal',3:'Reversable Defect'},

              'Heart Desease':{0:'Yes',1:'No'}

              },inplace=True)

sns.set(rc={'figure.figsize':(8,6)},style='white',palette="Reds_d")
ax = sns.heatmap(data2.corr().round(2),annot=True)
ax = sns.distplot(data2.Age)

pd.DataFrame(data2.Age.describe().round(2)).transpose()
ax = sns.kdeplot(data2[data2['Sex']=='Male']['Age'],shade=True)

ax = sns.kdeplot(data2[data2['Sex']=='Female']['Age'],shade=True)

ax = ax.set_xlabel('Age')

plt.legend(['Male','Female'])



data2.groupby('Sex').Age.describe().round(2)
ax = sns.boxplot(data=data2, x='Chest Pain Type', y='Age', width=0.5)

data2.groupby('Chest Pain Type').Age.describe().round(2)
ax = sns.lmplot(data=data2, x='Age', y='Resting Blood Pressure')

data2.groupby(pd.cut(data2["Age"],5))['Resting Blood Pressure'].describe()
ax = sns.lmplot(data=data2, x='Age', y='Cholesterol')

data2.groupby(pd.cut(data2["Age"],5))['Cholesterol'].describe()
ax = sns.boxplot(data=data2, x='Fasting Blood Sugar', y='Age', width=0.5)

data2.groupby('Fasting Blood Sugar').Age.describe().round(2)
ax = sns.boxplot(data=data2, x='Rest ECG', y='Age', width=0.5)

data2.groupby('Rest ECG').Age.describe().round(2)
ax = sns.lmplot(data=data2, x='Age', y='Max Heart Rate Achieved')

data2.groupby(pd.cut(data2["Age"],5))['Max Heart Rate Achieved'].describe()
ax = sns.boxplot(data=data2, x='Exercise Induced Angina', y='Age', width=0.5)

data2.groupby('Exercise Induced Angina').Age.describe().round(2)
ax = sns.lmplot(data=data2, x='Age', y='st Depression')

data2.groupby(pd.cut(data2["Age"],5))['st Depression'].describe()
ax = sns.boxplot(data=data2, x='st Slope', y='Age', width=0.5)

data2.groupby('st Slope').Age.describe().round(2)
ax = sns.boxplot(data=data2, x='Number Major Vessels', y='Age', width=0.5)

data2.groupby('Number Major Vessels').Age.describe().round(2)
ax = sns.boxplot(data=data2, x='Thalassemia', y='Age', width=0.5)

data2.groupby('Thalassemia').Age.describe().round(2)
ax = sns.boxplot(data=data2, x='Heart Desease', y='Age', width=0.5)

data2.groupby('Heart Desease').Age.describe().round(2)
ax = sns.countplot(data=data2,x="Sex",hue="Heart Desease")
print(round(len(data2[(data2['Sex']=='Male') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Sex']=='Male')]),2))

print(round(len(data2[(data2['Sex']=='Female') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Sex']=='Female')]),2))
ax = sns.kdeplot(data2[data2['Heart Desease']=='Yes']['Age'],shade=True)

ax = sns.kdeplot(data2[data2['Heart Desease']=='No']['Age'],shade=True)

ax = ax.set_xlabel('Age')

ax = plt.legend(['Yes','No'])



data2.groupby('Heart Desease').Age.describe().round(2)
ax = sns.countplot(data=data2,x="Chest Pain Type",hue="Heart Desease")
print(len(data2[(data2['Chest Pain Type']=='Typical Angina') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Chest Pain Type']=='Typical Angina')]))

print(len(data2[(data2['Chest Pain Type']=='Non-Anginal Pain') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Chest Pain Type']=='Non-Anginal Pain')]))

print(len(data2[(data2['Chest Pain Type']=='Atypical Angina') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Chest Pain Type']=='Atypical Angina')]))

print(len(data2[(data2['Chest Pain Type']=='Asymptomatic') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Chest Pain Type']=='Asymptomatic')]))
ax = sns.boxplot(data=data2,x='Heart Desease', y='Resting Blood Pressure')

data2.groupby('Heart Desease')['Resting Blood Pressure'].describe().round(2)
ax = sns.boxplot(data=data2,x='Heart Desease', y='Cholesterol')

data2.groupby('Heart Desease')['Cholesterol'].describe().round(2)
ax = sns.countplot(data=data2,x="Fasting Blood Sugar",hue="Heart Desease")
print(len(data2[(data2['Fasting Blood Sugar']=='> 120 mg/dl') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Fasting Blood Sugar']=='> 120 mg/dl')]))

print(len(data2[(data2['Fasting Blood Sugar']=='< 120 mg/dl') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Fasting Blood Sugar']=='< 120 mg/dl')]))
ax = sns.countplot(data=data2,x="Rest ECG",hue="Heart Desease")
print(len(data2[(data2['Rest ECG']=='Left Ventricular Hypertrophy') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Rest ECG']=='Left Ventricular Hypertrophy')]))

print(len(data2[(data2['Rest ECG']=='Normal') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Rest ECG']=='Normal')]))

print(len(data2[(data2['Rest ECG']=='ST-T wave abnormality') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Rest ECG']=='ST-T wave abnormality')]))
ax = sns.boxplot(data=data2,x='Heart Desease', y='Max Heart Rate Achieved')

data2.groupby('Heart Desease')['Max Heart Rate Achieved'].describe().round(2)
ax = sns.countplot(data=data2,x="Exercise Induced Angina",hue="Heart Desease")
print(len(data2[(data2['Exercise Induced Angina']=='No') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Exercise Induced Angina']=='No')]))

print(len(data2[(data2['Exercise Induced Angina']=='Yes') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Exercise Induced Angina']=='Yes')]))
ax = sns.boxplot(data=data2,x='Heart Desease', y='st Depression')

data2.groupby('Heart Desease')['st Depression'].describe().round(2)
ax = sns.countplot(data=data2,x="st Slope",hue="Heart Desease")
print(len(data2[(data2['st Slope']=='Downsloping') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['st Slope']=='Downsloping')]))

print(len(data2[(data2['st Slope']=='Upsloping') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['st Slope']=='Upsloping')]))

print(len(data2[(data2['st Slope']=='Flat') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['st Slope']=='Flat')]))
ax = sns.countplot(data=data2,x="Number Major Vessels",hue="Heart Desease")
print(len(data2[(data2['Number Major Vessels']==3) & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Number Major Vessels']==3)]))

print(len(data2[(data2['Number Major Vessels']==2) & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Number Major Vessels']==2)]))

print(len(data2[(data2['Number Major Vessels']==1) & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Number Major Vessels']==1)]))

print(len(data2[(data2['Number Major Vessels']==0) & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Number Major Vessels']==0)]))
ax = sns.countplot(data=data2,x="Thalassemia",hue="Heart Desease")
print(len(data2[(data2['Thalassemia']=='Fixed Defect') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Thalassemia']=='Fixed Defect')]))

print(len(data2[(data2['Thalassemia']=='Normal') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Thalassemia']=='Normal')]))

print(len(data2[(data2['Thalassemia']=='Reversable Defect') & (data2['Heart Desease']=='Yes')])/len(data2[(data2['Thalassemia']=='Reversable Defect')]))
data2.drop(['Resting Blood Pressure','Cholesterol', 'Fasting Blood Sugar'],axis=1,inplace=True)



y = data2['Heart Desease']



data2.drop(['Heart Desease'],axis=1, inplace=True)



x = data2
y.replace({'Heart Desease':{'Yes':0,'No':1}},inplace=True)
x = pd.get_dummies(x)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from sklearn.svm import LinearSVC



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)

scores = cross_val_score(clf, X_train, y_train, cv=5)

print("Accuracy: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))





clf.fit(X_train,y_train)

print("Test score: {:.2f}".format(accuracy_score(y_test,clf.predict(X_test))))

print("Cohen Kappa score: {:.2f}".format(cohen_kappa_score(y_test,clf.predict(X_test))))

ax = sns.heatmap(confusion_matrix(y_test,clf.predict(X_test)),annot=True)

ax= ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix')