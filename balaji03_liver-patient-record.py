import numpy as np 

import pandas as pd 

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

liver_patient = pd.read_csv("../input/indian-liver-patient-records/indian_liver_patient.csv")

liver_patient.head()
y = liver_patient['Dataset']

import seaborn as sns

liver_patient.isnull().sum()
liver = liver_patient.interpolate()

liver.isnull().sum()
liver['Gender'] = liver['Gender'].apply(lambda x: 1 if x =='Male' else 0)
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(liver['Dataset'])

ld,nld = liver['Dataset'].value_counts()

print('Number of people Diagnoised with Liver Disease:',ld)

print('Number of people with no Liver Disease:', nld)
Gender_comparision = liver[['Gender','Dataset']]

x = liver['Gender'][liver['Dataset']==1]

sns.countplot(x)

male,female = x.value_counts()

print("Number of Male's diagonised with Liver disease:",male)

print("Number of Female's diagonised with Liver Disease:",female)
liver[['Age','Gender','Dataset']].groupby(['Dataset','Gender'],as_index = False).mean().sort_values(by = 'Dataset',ascending = False)
correl = liver.drop(['Age'],axis = 1)

cor = correl.corr()

sns.heatmap(cor,annot = True, fmt = '.2f')
x = liver['Gender'][liver['Dataset']==1][liver['Albumin_and_Globulin_Ratio']>=0.9]

sns.countplot(x)

male,female = x.value_counts()

print('Number of Males diagonised with liver disease having Albumin and Globulin ratio > 0.9:',male)

print('Number of Females diagonised with liver disease having Albumin and Globulin ratio > 0.9:',female)
liver[['Albumin','Albumin_and_Globulin_Ratio', 'Dataset']].groupby('Dataset',as_index= False).mean()
liver_dis = liver.drop(['Dataset'],axis = 1)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,accuracy_score

from sklearn.metrics import confusion_matrix, classification_report
logistic_model = LogisticRegression()

train_x,test_x,train_y,test_y = train_test_split(liver_dis,y,test_size = 0.3,random_state = 123)

logistic_model.fit(train_x,train_y)

prediction = logistic_model.predict(test_x)

accuracy = accuracy_score(test_y,prediction)*100

mea = mean_absolute_error(test_y,prediction)

print(accuracy,mea)