import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(color_codes = True)
pima = pd.read_csv('../input/diabetes.csv')
pima
pima.head()
pima.tail()
pima.shape
pima.info()
pima.describe()
pima.describe().T
pima['Outcome'].value_counts()
plt.figure(figsize=(15,6))
sns.countplot(pima['Outcome'])
plt.ylabel('Number of People')
pima[pima['Glucose'] == 0]
missingGlucose = pima[pima['Glucose'] == 0].shape[0]
print ("Number of zeros in colum Glucose: ", missingGlucose)
missingBP = pima[pima['BloodPressure'] == 0].shape[0]
print ("Number of zeros in colum BloodPressure: ", missingBP)
missingST = pima[pima['SkinThickness'] == 0].shape[0]
print ("Number of zeros in colum SkinThickness: ", missingST)
missingInsulin = pima[pima['Insulin'] == 0].shape[0]
print ("Number of zeros in colum Insulin: ", missingInsulin)
missingBMI = pima[pima['BMI'] == 0].shape[0]
print ("Number of zeros in colum BMI: ", missingBMI)
pima_copy = pima.copy(deep = True)
pima_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = pima_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(pima_copy.isnull().sum())
pima.hist(figsize = (20,20))
plt.figure(figsize=(15,6))
sns.distplot(pima['Glucose'], kde = True, rug = True)
plt.figure(figsize=(15,6))
sns.boxplot(pima['Glucose'])
plt.figure(figsize=(15,6))
sns.distplot(pima['BloodPressure'], kde = True, rug = True)
plt.figure(figsize=(15,6))
sns.boxplot(pima['BloodPressure'])
plt.figure(figsize=(15,6))
sns.distplot(pima['SkinThickness'], kde = True, rug = True)
plt.figure(figsize=(15,6))
sns.boxplot(pima['SkinThickness'])
plt.figure(figsize=(15,6))
sns.distplot(pima['Insulin'], kde = True, rug = True)
plt.figure(figsize=(15,6))
sns.boxplot(pima['Insulin'])
plt.figure(figsize=(15,6))
sns.distplot(pima['BMI'], kde = True, rug = True)
plt.figure(figsize=(15,6))
sns.boxplot(pima['BMI'])
pima_copy['Glucose'].fillna(pima_copy['Glucose'].mean(), inplace = True)
pima_copy['BloodPressure'].fillna(pima_copy['BloodPressure'].mean(), inplace = True)
pima_copy['SkinThickness'].fillna(pima_copy['SkinThickness'].median(), inplace = True)
pima_copy['Insulin'].fillna(pima_copy['Insulin'].median(), inplace = True)
pima_copy['BMI'].fillna(pima_copy['BMI'].median(), inplace = True)
print(pima_copy.isnull().sum())
pima_copy.describe().T
plt.figure(figsize=(15,6))
sns.distplot(pima_copy['Pregnancies'], bins=20, rug = True)
plt.ylabel('Number of People')
print("Average of children had by Pima woman: ", pima_copy['Pregnancies'].mean())
pima_copy['Pregnancies'].median()
preg = pima_copy[pima_copy['Pregnancies'] >= 1].shape[0]
print('Number of Pima Woman who had children: ', preg)
notPreg = pima_copy[pima_copy['Pregnancies'] == 0].shape[0]
print('Number of Pima woman who did not have children: ', notPreg)
pregPlusDiabetes = pima_copy[(pima_copy['Pregnancies'] >= 1) & (pima_copy['Outcome'] == 1)].shape[0]
print('Number of woman who have children and are diabetic: ',pregPlusDiabetes)
pregPlusNotDiabetes = pima_copy[(pima_copy['Pregnancies'] >= 1) & (pima_copy['Outcome'] == 0)].shape[0]
print('Number of woman who have children and are not diabetic: ',pregPlusNotDiabetes)
notPregPlusDiabetes = pima_copy[(pima_copy['Pregnancies'] == 0) & (pima_copy['Outcome'] == 1)].shape[0]
print('Number of woman who do not have children and are diabetic: ',notPregPlusDiabetes)
notPregPlusNotDiabetes = pima_copy[(pima_copy['Pregnancies'] == 0) & (pima_copy['Outcome'] == 0)].shape[0]
print('Number of woman who do not have children and are not diabetic: ',notPregPlusNotDiabetes)
plt.figure(figsize=(15,6))
sns.distplot(pima_copy['Glucose'], bins=20, rug = True)
plt.figure(figsize=(15,6))
sns.distplot(pima_copy['BloodPressure'], bins=20, rug = True)
plt.figure(figsize=(15,6))
sns.distplot(pima_copy['SkinThickness'], bins=20, rug = True)
plt.figure(figsize=(15,6))
sns.distplot(pima_copy['Insulin'], bins = 50)
plt.figure(figsize=(15,6))
sns.distplot(pima_copy['BMI'], bins = 50)
print('Average BMI: ', pima_copy['BMI'].mean())
plt.figure(figsize=(15,6))
sns.distplot(pima_copy['DiabetesPedigreeFunction'], bins = 50)
plt.figure(figsize=(15,6))
sns.distplot(pima_copy['Age'], bins = 50)
print("Minimum age: ",pima_copy['Age'].min())
print("Maximum age: ",pima_copy['Age'].max())
corr = pima_copy.corr()
corr
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap = 'plasma', vmin = -1, vmax = 1)
print('Average Glucose for Pima woman who has diabetes: ', pima_copy[pima_copy['Outcome'] == 1]['Glucose'].mean())
print('Average Glucose for Pima woman who does not have diabetes: ', pima_copy[pima_copy['Outcome'] == 0]['Glucose'].mean())
sns.boxplot(pima['Outcome'],pima['Glucose'])
print('Average BMI for Pima woman who has diabetes: ', pima_copy[pima_copy['Outcome'] == 1]['BMI'].mean())
print('Average BMI for Pima woman who does not have diabetes: ', pima_copy[pima_copy['Outcome'] == 0]['BMI'].mean())
plt.figure(figsize=(15,6))
sns.boxplot(pima_copy['Outcome'],pima_copy['Age'])
oneOutcome = pima_copy[pima_copy['Outcome'] == 1]
print("Minimum age of Pima woman who has Diabetes: ",oneOutcome['Age'].min())
print("Maximum age of Pima woman who has Diabetes: ",oneOutcome['Age'].max())
zeroOutcome = pima_copy[pima_copy['Outcome'] == 0]
print("Minimum age of Pima woman who does not have Diabetes: ",zeroOutcome['Age'].min())
zeroOutcome = pima_copy[pima_copy['Outcome'] == 0]
print("Maximum age of Pima woman who does not have Diabetes: ",zeroOutcome['Age'].max())
print('Average Age of Pima woman who has diabetes: ',pima_copy[pima_copy['Outcome'] == 1]['Age'].mean())
print('Average Age of Pima woman who does not have diabetes: ',pima_copy[pima_copy['Outcome'] == 0]['Age'].mean())
plt.figure(figsize=(15,6))
sns.countplot(x = 'Pregnancies', hue = 'Outcome', data = pima_copy)
print('Average Skin Thickness of Pima woman who has diabetes: ', pima_copy[pima_copy['Outcome'] == 1]['SkinThickness'].mean())
print('Average Skin Thickness of Pima woman who does not have diabetes: ', pima_copy[pima_copy['Outcome'] == 0]['SkinThickness'].mean())
print('Average Insulin of Pima woman who has diabetes: ', pima_copy[pima_copy['Outcome'] == 1]['Insulin'].mean())
print('Average Insulin of Pima woman who does not have diabetes: ', pima_copy[pima_copy['Outcome'] == 0]['Insulin'].mean())
sns.pairplot(pima_copy)
pima_copy.hist(figsize = (20,20))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
x = pima_copy.drop('Outcome', axis  = 1)
y = pima_copy['Outcome']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 17)
mdl = LogisticRegression()
fit = mdl.fit(X_train, y_train)
pred = mdl.predict(X_test)
confusion = metrics.confusion_matrix(y_test,pred)
confusion
label = ["No (0)"," Yes (1)"]
sns.heatmap(confusion, annot=True, xticklabels=label, yticklabels=label)
metrics.accuracy_score(y_test,pred)
metrics.precision_score(y_test,pred)
x = pima.drop('Outcome', axis  = 1)
y = pima['Outcome']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 17)
mdl = LogisticRegression()
fit = mdl.fit(X_train, y_train)
pred = mdl.predict(X_test)
pred
metrics.accuracy_score(y_test,pred)
metrics.precision_score(y_test,pred)
