# Here we load the necessary Libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
#Using Powerful Pandas Library We can read and Manipulate the data.

data=pd.read_csv("../input/heart.csv")
#Displaying few rows

data.head()
data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

#changing the Categorical Variables



data.sex[data.sex==0]='Female'

data.sex[data.sex==1]='Male'





data.chest_pain_type[data.chest_pain_type==0]='typical angina'

data.chest_pain_type[data.chest_pain_type== 1] = 'atypical angina'

data.chest_pain_type[data.chest_pain_type== 2] = 'non-anginal pain'

data.chest_pain_type[data.chest_pain_type== 3] = 'asymptomatic'



data.rest_ecg[data.rest_ecg == 0] = 'normal'

data.rest_ecg[data.rest_ecg == 1] = 'ST-T wave abnormality'

data.rest_ecg[data.rest_ecg == 2] = 'left ventricular hypertrophy'



data.fasting_blood_sugar[data.fasting_blood_sugar == 0] = 'lower than 120mg/ml'

data.fasting_blood_sugar[data.fasting_blood_sugar == 1] = 'greater than 120mg/ml'





data.exercise_induced_angina[data.exercise_induced_angina == 0] = 'no'

data.exercise_induced_angina[data.exercise_induced_angina == 1] = 'yes'





data.st_slope[data.st_slope== 0] = 'upsloping'

data.st_slope[data.st_slope== 1] = 'flat'

data.st_slope[data.st_slope== 2] = 'downsloping'



data.thalassemia[data.thalassemia == 1] = 'normal'

data.thalassemia[data.thalassemia ==  2] = 'fixed defect'

data.thalassemia[data.thalassemia == 3] = 'reversable defect'
data.head()
data.isnull().sum()
data.dtypes
data.target=data.target.astype('object')
# Target Vs Sex-->Count Plot

sns.countplot(x=data.sex,hue=data.target,palette='Set1')

plt.title("Heart Disease(Male,Female)")

plt.ylabel("No of patients")

plt.legend(['Normal','Heart patients'],loc=1)
#Target vs chest pain

sns.countplot(x=data.chest_pain_type,hue=data.target,palette='Set1')

plt.ylabel("No of patients")

plt.legend(['Normal','Heart patients'],loc=2)

plt.title("Chest Pain Type vs Heart disease")

plt.xlabel("Chest Pain Type")
#Target VS Fasting Blood Sugar

sns.countplot(x=data.fasting_blood_sugar,hue=data.target,palette='Set1')

plt.ylabel("No of patients")

plt.legend(['Normal','Heart patients'],loc=2)

plt.title("Heart Disease due to Blood Sugar")

plt.xlabel("Fasting Blood Sugar")
#Target Vs RestECG

plt.figure(figsize=(8.4,6))

sns.countplot(x=data.rest_ecg,hue=data.target,palette='Set1')

plt.legend(['Normal','Heart patients'],loc=1)

plt.title("Resting electrocardiographic measurement vs Target  ")

plt.xlabel("Resting electrocardiographic measurement")

#Target vs Slope Segment

plt.figure(figsize=(8.4,6))

sns.countplot(x=data.st_slope,hue=data.target,palette='Set1')

plt.title("Slope Segment Vs Target")

plt.xlabel("Slope Segmet")

plt.legend(['Normal','Heart patients'],loc=1)

#Target vs Blood disorder

plt.figure(figsize=(8.4,6))

sns.countplot(x=data.thalassemia,hue=data.target,palette='Set1')

plt.legend(['Normal','Heart patients'],loc=1)

plt.title("Target vs Blood Disorder")

plt.xlabel("Blood Disorder")
#Target Vs Exercise Induced angina

plt.figure(figsize=(8.4,6))

sns.countplot(x=data.exercise_induced_angina,hue=data.target,palette='Set1')

plt.xlabel("Exercie Induced Angina")

plt.title("Target Vs Exercise Induced Angina")

plt.legend(['Normal','Heart patients'],loc=1)

# Target VS age

plt.figure(figsize=(12,6))

sns.countplot(x=data.age,hue=data.target,palette="rainbow")

plt.legend(['Normal','Heart patients'],loc=1)

plt.title("Age VS Target")
data['Age_grp']=pd.cut(x=data.age,bins=range(25,85,10))#creating Age Grp Columns

plt.figure(figsize=(8.4,6))

sns.countplot(x=data.Age_grp,hue=data.target,palette='Set1')

plt.legend(['Normal','Heart patients'],loc=2)

plt.title("Age_grp VS Target")
#target Vs Blood pressure

plot = data[data.target == 1].resting_blood_pressure.value_counts().sort_index().plot(kind = "bar", figsize=(15,4), fontsize = 15)

plt.title("Blood pessure Level in heart Patients")
#Age vs Chol level

plt.figure(figsize=(10,8))

sns.lineplot(x=data.age,y=data.cholesterol,hue=data.target)

plt.legend(['Normal','Heart patients'],loc=1)

plt.title("Chol level in Different Age Groups")

data.target=data.target.astype('int')

plt.figure(figsize=(15,10))

sns.heatmap(data.corr(),annot=True,cmap='YlGnBu')
data=pd.get_dummies(data,drop_first=True)


data.head()
plt.figure(figsize=(15,10))

sns.heatmap(data.corr(),annot=True,cmap='YlGnBu')
x=data.drop('target',axis=1)

y=data['target'].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()

lg.fit(x_train, y_train)

y_pred=lg.predict(x_test)
from sklearn.metrics import confusion_matrix,r2_score,accuracy_score

cmat = confusion_matrix(y_pred,y_test)

sns.heatmap(cmat, annot=True)
print('Accurancy:',accuracy_score(y_test, y_pred))

print("Logistic TRAIN score with ",(lg.score(x_train, y_train)))

print("Logistic TEST score with ",(lg.score(x_test, y_test)))