from IPython.display import YouTubeVideo

YouTubeVideo("pN4HqWRybwk")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(color_codes = True)





#ignore warning messages 

import warnings

warnings.filterwarnings('ignore')
pima = pd.read_csv('../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv')
pima
pima.head()
pima.tail()
pima.shape
pima.dtypes
pima.info()
pima.isnull().values.any()
pima.describe().T
pima['class'].value_counts()
plt.figure(figsize=(10,7))

sns.set(font_scale = 1.5)

sns.countplot(x = 'class', data=pima, palette="Set2")

plt.ylabel('Number of People')
plt.figure(figsize=(8,8))

pieC = pima['class'].value_counts()

explode = (0.05, 0)

colors = ['moccasin', 'coral']

labels = ['0 - Non Diabetic', '1 - Diabetic']

sns.set(font_scale = 1.5)

plt.pie(pieC, labels = ('0 - Non Diabetic', '1 - Diabetic'), autopct = "%.2f%%", explode = explode, colors = colors)

plt.legend(labels, loc = 'lower left')
pima[pima['Plas'] == 0]
missingPlas = pima[pima['Plas'] == 0].shape[0]

print ("Number of zeros in variable Plas (Glucose): ", missingPlas)
missingPres = pima[pima['Pres'] == 0].shape[0]

print ("Number of zeros in variable Pres (BloodPressure): ", missingPres)
missingSkin = pima[pima['skin'] == 0].shape[0]

print ("Number of zeros in variable Skin (SkinThickness): ", missingSkin)
missingTest = pima[pima['test'] == 0].shape[0]

print ("Number of zeros in variable Test (Insulin): ", missingTest)
missingMass = pima[pima['mass'] == 0].shape[0]

print ("Number of zeros in variable Mass (BMI): ", missingMass)
pima_copy = pima.copy(deep = True)
pima_copy[['Plas','Pres','skin','test','mass']] = pima_copy[['Plas','Pres','skin','test','mass']].replace(0,np.NaN)

print(pima_copy.isnull().sum())
pima.hist(figsize = (20,16),grid=True)
pima.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(15,15))

sns.set(font_scale = 1.5)
fig, ax = plt.subplots(4,2, figsize=(16,16))

sns.set(font_scale = 1)

sns.distplot(pima.Plas, ax = ax[0,0], color = 'orange')

sns.distplot(pima.Preg, ax = ax[0,1], color = 'red')

sns.distplot(pima.Pres, ax = ax[1,0], color = 'seagreen')

sns.distplot(pima.age, ax = ax[1,1], color = 'purple')

sns.distplot(pima.mass, ax = ax[2,0], color = 'deeppink')

sns.distplot(pima.pedi, ax = ax[2,1], color = 'brown')

sns.distplot(pima.skin, ax = ax[3,0], color = 'royalblue')

sns.distplot(pima.test, ax = ax[3,1], color = 'coral')
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.distplot(pima['Plas'], kde = True, rug = True, color = 'orange')
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.boxplot(pima.Plas, color = 'orange')
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.distplot(pima['Pres'], kde = True, rug = True, color = 'seagreen')
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.boxplot(pima.Pres, color = 'seagreen')
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.distplot(pima['skin'], kde = True, rug = True, color = 'royalblue')
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.boxplot(pima.skin, color = 'royalblue')
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.distplot(pima['test'], kde = True, rug = True, color = 'coral')
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.boxplot(pima.test, color = 'coral')
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.distplot(pima['mass'], kde = True, rug = True, color = 'deeppink')
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.boxplot(pima.mass, color = 'deeppink')
pima_copy['Plas'].fillna(pima_copy['Plas'].mean(), inplace = True)
pima_copy['Pres'].fillna(pima_copy['Pres'].mean(), inplace = True)
pima_copy['skin'].fillna(pima_copy['skin'].median(), inplace = True)
pima_copy['test'].fillna(pima_copy['test'].median(), inplace = True)
pima_copy['mass'].fillna(pima_copy['mass'].median(), inplace = True)
print(pima_copy.isnull().sum())
pima_copy.describe().T
pima_copy.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(15,15))

sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.countplot(pima_copy['Preg'])

plt.ylabel('Number of People')
print("Average number of children had by Pima woman: ", pima_copy['Preg'].mean())
pima_copy['Preg'].median()
preg = pima_copy[pima_copy['Preg'] >= 1].shape[0]

print('Number of Pima Woman who had children: ', preg)
notPreg = pima_copy[pima_copy['Preg'] == 0].shape[0]

print('Number of Pima woman who did not have children: ', notPreg)
pregPlusDiabetes = pima_copy[(pima_copy['Preg'] >= 1) & (pima_copy['class'] == 1)].shape[0]

print('Number of woman who have children and are diabetic: ',pregPlusDiabetes)
pregPlusNotDiabetes = pima_copy[(pima_copy['Preg'] >= 1) & (pima_copy['class'] == 0)].shape[0]

print('Number of woman who have children and are not diabetic: ',pregPlusNotDiabetes)
notPregPlusDiabetes = pima_copy[(pima_copy['Preg'] == 0) & (pima_copy['class'] == 1)].shape[0]

print('Number of woman who do not have children and are diabetic: ',notPregPlusDiabetes)
notPregPlusNotDiabetes = pima_copy[(pima_copy['Preg'] == 0) & (pima_copy['class'] == 0)].shape[0]

print('Number of woman who do not have children and are not diabetic: ',notPregPlusNotDiabetes)
corr = pima_copy.corr()

corr
plt.figure(figsize=(15,10))

sns.set(font_scale = 1.5)

sns.heatmap(corr, annot = True, cmap = 'plasma', vmin = -1, vmax = 1, linecolor='white', linewidths= 1)
print('Average Glucose for Pima woman who has diabetes: ', pima_copy[pima_copy['class'] == 1]['Plas'].mean())
print('Average Glucose for Pima woman who does not have diabetes: ', pima_copy[pima_copy['class'] == 0]['Plas'].mean())
plt.figure(figsize=(15,7))

sns.boxplot(pima_copy['class'],pima_copy['Plas'], palette="Set2")

sns.set(font_scale = 1.5)
print('Average BMI for Pima woman who has diabetes: ', pima_copy[pima_copy['class'] == 1]['mass'].mean())
print('Average BMI for Pima woman who does not have diabetes: ', pima_copy[pima_copy['class'] == 0]['mass'].mean())
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.boxplot(pima_copy['class'],pima_copy['age'], palette = "Set3")
oneOutcome = pima_copy[pima_copy['class'] == 1]

print("Minimum age of Pima woman who has Diabetes: ",oneOutcome['age'].min())
print("Maximum age of Pima woman who has Diabetes: ",oneOutcome['age'].max())
zeroOutcome = pima_copy[pima_copy['class'] == 0]

print("Minimum age of Pima woman who does not have Diabetes: ",zeroOutcome['age'].min())
zeroOutcome = pima_copy[pima_copy['class'] == 0]

print("Maximum age of Pima woman who does not have Diabetes: ",zeroOutcome['age'].max())
print('Average Age of Pima woman who has diabetes: ',pima_copy[pima_copy['class'] == 1]['age'].mean())
print('Average Age of Pima woman who does not have diabetes: ',pima_copy[pima_copy['class'] == 0]['age'].mean())
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

sns.countplot(x = 'Preg', hue = 'class', data = pima_copy, palette = 'Set2')
print('Average Skin Thickness of Pima woman who has diabetes: ', pima_copy[pima_copy['class'] == 1]['skin'].mean())
print('Average Skin Thickness of Pima woman who does not have diabetes: ', pima_copy[pima_copy['class'] == 0]['skin'].mean())
print('Average Insulin of Pima woman who has diabetes: ', pima_copy[pima_copy['class'] == 1]['test'].mean())
print('Average Insulin of Pima woman who does not have diabetes: ', pima_copy[pima_copy['class'] == 0]['test'].mean())
sns.set(font_scale = 1.5)

sns.pairplot(data = pima_copy, hue = 'class', diag_kind = 'kde', palette = 'Set2')
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

X = pima_copy.drop('class', axis  = 1)

y = pima_copy['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 17)
print(X_train.shape)

print(X_test.shape)

print(y_train.size)

print(y_test.size)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
confusion = metrics.confusion_matrix(y_test,y_pred)

confusion
ylabel = ["Actual [Non-Diab]","Actual [Diab]"]

xlabel = ["Pred [Non-Diab]","Pred [Diab]"]

#sns.set(font_scale = 1.5)

plt.figure(figsize=(15,6))

sns.heatmap(confusion, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)
print('Accuracy of Logistic Regression is: ', model.score(X_test,y_test) * 100,'%')
print(classification_report(y_test,y_pred))
TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]
# print ('Precision: ', metrics.precision_score(y_test,y_pred))

Precision = TP / ( TP + FP )

print ('Precision: ', Precision)
Recall = TP / ( TP + FN )

print ('Recall: ', Recall)
pima_copy['class'].value_counts()
metrics.f1_score(y_test, y_pred)
Specificity = TN / ( TN + FP )

print ('Specificity: ', Specificity)
Sensitivity = TP / ( TP + FN )

print ('Sensitivity: ', Sensitivity)
Roc_Auc = metrics.roc_auc_score(y_test, y_pred)

print ('Roc Auc Score: ', Roc_Auc)
y_pred_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

plt.plot(fpr, tpr)

plt.title('ROC Curve for Logistic Regression Diabetes Classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

plt.xlim(0.0, 1.0)

plt.ylim(0.0, 1.0)

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
nbModel = GaussianNB()
nbModel.fit(X_train, y_train)
nb_y_pred = nbModel.predict(X_test)
nbConfusion = metrics.confusion_matrix(y_test, nb_y_pred)

nbConfusion
ylabel = ["Actual [Non-Diab]","Actual [Diab]"]

xlabel = ["Pred [Non-Diab]","Pred [Diab]"]

#sns.set(font_scale = 1.5)

plt.figure(figsize=(15,6))

sns.heatmap(nbConfusion, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)
print('Accuracy of Naive Bayes Classifier is: ', nbModel.score(X_test,y_test) * 100,'%')
print(classification_report(y_test, nb_y_pred))
nb_TP = nbConfusion[1, 1]

nb_TN = nbConfusion[0, 0]

nb_FP = nbConfusion[0, 1]

nb_FN = nbConfusion[1, 0]
# print ('Precision: ', metrics.precision_score(y_test, nb_y_pred))

nb_Precision = nb_TP / ( nb_TP + nb_FP)

print ('Precision: ', nb_Precision)
nb_Recall = nb_TP / ( nb_TP + nb_FN )

print ('Recall: ', nb_Recall)
pima_copy['class'].value_counts()
metrics.f1_score(y_test, nb_y_pred)
nb_Specificity = nb_TN / ( nb_TN + nb_FP )

print ('Specificity: ', nb_Specificity)
nb_Sensitivity = nb_TP / ( nb_TP + nb_FN )

print ('Sensitivity: ', nb_Sensitivity)
nb_Roc_Auc = metrics.roc_auc_score(y_test,nb_y_pred)

print ('Roc Auc Score: ', nb_Roc_Auc)
nb_y_pred_prob = nbModel.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, nb_y_pred_prob)
plt.figure(figsize=(15,6))

sns.set(font_scale = 1.5)

plt.plot(fpr, tpr)

plt.title('ROC Curve for Naive Bayes Diabetes Classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

plt.xlim(0.0, 1.0)

plt.ylim(0.0, 1.0)

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')