import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')


import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

patients=pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')
patients.head()
patients.shape
patients.info()
patients.describe().T
patients['Gender']=patients['Gender'].apply(lambda x:1 if x=='Male' else 0)
patients.head()
patients['Gender'].value_counts().plot.bar(color='peachpuff')

plt.show()
patients['Dataset'].value_counts().plot.bar(color='blue')

plt.show()
patients.isnull().sum()
patients['Albumin_and_Globulin_Ratio'].mean()
patients=patients.fillna(0.94)
patients.isnull().sum()
sns.set_style('darkgrid')

plt.figure(figsize=(25,10))

patients['Age'].value_counts().plot.bar(color='darkviolet')

plt.show()
plt.rcParams['figure.figsize']=(10,10)

sns.pairplot(patients,hue='Gender')

plt.show()
sns.pairplot(patients)

plt.show()
f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x="Albumin", y="Albumin_and_Globulin_Ratio",color='mediumspringgreen',data=patients);

plt.show()
plt.figure(figsize=(8,6))

patients.groupby('Gender').sum()["Total_Protiens"].plot.bar(color='coral')

plt.show()
plt.figure(figsize=(8,6))

patients.groupby('Gender').sum()['Albumin'].plot.bar(color='midnightblue')

plt.show()
plt.figure(figsize=(8,6))

patients.groupby('Gender').sum()['Total_Bilirubin'].plot.bar(color='fuchsia')

plt.show()
corr=patients.corr()
plt.figure(figsize=(20,10)) 

sns.heatmap(corr,cmap="Greens",annot=True)

plt.show()
from sklearn.model_selection import train_test_split
patients.columns
X=patients[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',

       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',

       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',

       'Albumin_and_Globulin_Ratio']]

y=patients['Dataset']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5,random_state=42)

logmodel = LogisticRegression(C=1, penalty='l1')

results = cross_val_score(logmodel, X_train,y_train,cv = kfold)

print(results)

print("Accuracy:",results.mean()*100)