import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import eli5
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from eli5.sklearn import PermutationImportance
from pandas.plotting import scatter_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
data = pd.read_csv("../input/heart-disease-uci/heart.csv") 
data.head()
#age distribution
age = data['age']
age_bins = [x*5 for x in range(3,18)]
plt.xlabel('age')
plt.xticks(age_bins)

plt.hist(age,bins=age_bins)
plt.title('ages distribution')
#class 0 = blue = no disease
#class 1 = red = heart disease

grouped = data.groupby('target')

class0 = grouped.get_group(0)
class1 = grouped.get_group(1)

#correlation of features
attributes = ["trestbps","thalach","oldpeak","age", "chol"]
scatter_matrix(data[attributes], figsize=(15,15))


figure1 = plt.figure()
ax = figure1.gca(projection='3d')

ax.scatter(class0['chol'].values, class0['trestbps'].values, class0['oldpeak'].values, color = 'b')
ax.scatter(class1['chol'].values, class1['trestbps'].values, class1['oldpeak'].values, color = 'r')

ax.set_xlabel('chol')
ax.set_ylabel('trestbps')
ax.set_zlabel('oldpeak')


figure2 = plt.figure()
ax = figure2.gca(projection='3d')

ax.scatter(class0['trestbps'].values, class0['exang'].values, class0['oldpeak'].values, color = 'b')
ax.scatter(class1['trestbps'].values, class1['exang'].values, class1['oldpeak'].values, color = 'r')

ax.set_xlabel('trestbps')
ax.set_ylabel('exang')
ax.set_zlabel('oldpeak')

plt.show()
exang0 = data.loc[data['exang'] == 0]
exang1 = data.loc[data['exang'] == 1]

print('number of heart disease when exang = 0 :', exang0[exang0['target'] == 1].values.shape[0])
print('number of heart disease when exang = 1 :', exang1[exang1['target'] == 1].values.shape[0])

print("people who don't have exang that develop an heart disease = ",round(100*exang0[exang0['target'] == 1].values.shape[0]/exang0.values.shape[0],0), '%')
print("people who have exang that develop an heart disease = ",round(100*exang1[exang1['target'] == 1].values.shape[0]/exang1.values.shape[0],0), '%')


#disease repartition depending on gender
female = data.loc[data['sex'] == 0]
male = data.loc[data['sex'] == 1]

print('number of men in sample = ', male.values.shape[0])
print('number of women in sample = ', female.values.shape[0])
print('men that develop heart disease = ', round(100*male[male['target'] == 1].values.shape[0]/male.values.shape[0],0), '%')
print('women that develop heart disease = ', round(100*female[female['target'] == 1].values.shape[0]/female.values.shape[0],0), '%')
data_copy = data.copy()
data_copy.drop(columns=['target'],inplace=True)

matrix_corr = data_copy.corr()
fig, ax = plt.subplots(figsize=(15,15))
ax.matshow(matrix_corr, cmap='coolwarm')

labels = list(data_copy.columns)

ax.xaxis.set(ticks=range(0, data_copy.columns.size), ticklabels=labels)
ax.yaxis.set(ticks=range(0, data_copy.columns.size), ticklabels=labels)

for (i, j), z in np.ndenumerate(matrix_corr):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.show()
print('number of elements', data.shape[0])
print('number of elements in class 1 =', class0.values.shape[0])
print('number of elements in class 0 =', class1.values.shape[0])
data.isnull().sum()
print('average of age for class 0 (no disease) = ', round(class0['age'].mean(),1))
print('average of age for class 1 (disease) = ', round(class1['age'].mean(),1))

print('average oldpeak for class 0 (no disease) = ', round(class0['oldpeak'].mean(),1))
print('average oldpeak for class 1 (disease) = ', round(class1['oldpeak'].mean(),1))
#oldpeak = ST depression induced by exercise relative to rest

print('average trestbps for class 0 (no disease) = ', round(class0['trestbps'].mean(),1))
print('average trestbps for class 1 (disease) = ', round(class1['trestbps'].mean(),1))
#trestbps = resting blood pressure (in mm Hg on admission to the hospital)
#preparation of dataset
#random oversampling on class 0
df_minority = data[data.target==0]
df_majority = data[data.target==1]

df_class_0_over = df_minority.sample(df_majority.values.shape[0],replace=True)
df = pd.concat([df_class_0_over, df_majority], axis=0)

print('Random over-sampling:')
print(df.target.value_counts())

#only take numerical columns
data_ = df[['age','chol','trestbps','thalach','oldpeak','ca']]

y = df[['target']]
y = y.values.ravel()
X = data_.values
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_log = LogisticRegression(random_state=0).fit(X_train, y_train)

print(clf_log.score(X_test,y_test))

cm = confusion_matrix(y_test, clf_log.predict(X_test))

fig = plt.figure()
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')


#sensibilité 
#sensibilité = vrai positif/(vrai positif + faux négatif)
#=probabilité que le test soit positif si la maladie est présente
sensibility = cm[1,1]/(cm[1,1]+cm[1,0])
print('sensibility = ',sensibility)
#spécificité
#sensibilité = vrai négatif/(vrai négatif + faux positif)
#=probabilité d'obtenir un test négatif chez les non-malades
specificity = cm[0,0]/(cm[0,0]+cm[0,1])
print('specificity = ', specificity)
plt.show()

#importance des features
fig5, ax = plt.subplots()
bins = [x for x in range(0,data_.columns.size)]
values = (clf_log.coef_).flatten()
plt.bar(bins,values,align='center')
ax.set_xticks(np.arange(data_.columns.size))
ax.set_xticklabels(list(data_.columns),rotation = 45)
plt.show()

perm = PermutationImportance(clf_log, random_state=0).fit(X_test, y_test)
features_n = list(data_.columns)
eli5.show_weights(perm,feature_names = features_n)

#only take categorical columns
data_ = df[['sex','cp','fbs','restecg','exang','thal','slope']]
X = data_.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimators = [5,10,15,20,30,40,50,60,70,80,90]
precisions = []
for i in estimators:
    classifier = RandomForestClassifier(n_estimators=i, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    precisions.append(accuracy_score(y_test, y_pred))
    

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

figure10 = plt.figure()
plt.plot(estimators,precisions)
plt.show()

cm = confusion_matrix(y_test, classifier.predict(X_test))
#sensibilité 
#sensibilité = vrai positif/(vrai positif + faux négatif)
#=probabilité que le test soit positif si la maladie est présente
sensibility = cm[1,1]/(cm[1,1]+cm[1,0])
print('sensibility = ',sensibility)
#spécificité
#sensibilité = vrai négatif/(vrai négatif + faux positif)
#=probabilité d'obtenir un test négatif chez les non-malades
specificity = cm[0,0]/(cm[0,0]+cm[0,1])
print('specificity = ', specificity)
plt.show()
fig = plt.figure()
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
X = df[['sex','cp','fbs','restecg','exang','thal','slope','age','chol','trestbps','thalach','oldpeak','ca']]
X = X.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

forest_only = RandomForestClassifier(n_estimators=80, random_state=0)
forest_only.fit(X_train, y_train)
y_pred = forest_only.predict(X_test)

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
#sensibilité 
#sensibilité = vrai positif/(vrai positif + faux négatif)
#=probabilité que le test soit positif si la maladie est présente
sensibility = cm[1,1]/(cm[1,1]+cm[1,0])
print('sensibility = ',sensibility)
#spécificité
#sensibilité = vrai négatif/(vrai négatif + faux positif)
#=probabilité d'obtenir un test négatif chez les non-malades
specificity = cm[0,0]/(cm[0,0]+cm[0,1])
print('specificity = ', specificity)

fig = plt.figure()
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

#1:logistic regression
y = df[['target']]
y = y.values.ravel()
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_log = X_train[['age','chol','trestbps','thalach','oldpeak','ca']]
X_train_log = preprocessing.scale(X_train_log)
clf_log = LogisticRegression(random_state=0).fit(X_train_log, y_train)

logistic_predicton = clf_log.predict(X_train_log)  
log_pred = pd.DataFrame({'log_pred': logistic_predicton},index=X_train.index)


#2:Random forest
X_train_forest = X_train[['sex','cp','fbs','restecg','exang','thal','slope']]
X_train_forest['log_pred'] = log_pred

classifier_forest = RandomForestClassifier(n_estimators=80, random_state=0)
classifier_forest.fit(X_train_forest, y_train)



#test
X_test_log = X_test[['age','chol','trestbps','thalach','oldpeak','ca']]
X_test_log = preprocessing.scale(X_test_log)
logistic_pred = clf_log.predict(X_test_log)
log_pred_test = pd.DataFrame({'log_pred': logistic_pred},index=X_test.index)

X_test_forest = X_test[['sex','cp','fbs','restecg','exang','thal','slope']]
X_test_forest['log_pred'] = log_pred_test

y_pred = classifier_forest.predict(X_test_forest)

 
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

#sensibilité 
#sensibilité = vrai positif/(vrai positif + faux négatif)
#=probabilité que le test soit positif si la maladie est présente
sensibility = cm[1,1]/(cm[1,1]+cm[1,0])
print('sensibility = ',sensibility)
#spécificité
#sensibilité = vrai négatif/(vrai négatif + faux positif)
#=probabilité d'obtenir un test négatif chez les non-malades
specificity = cm[0,0]/(cm[0,0]+cm[0,1])
print('specificity = ', specificity)
plt.show()
fig = plt.figure()
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')