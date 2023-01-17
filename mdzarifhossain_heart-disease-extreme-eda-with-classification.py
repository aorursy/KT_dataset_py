import matplotlib.pyplot as plt #for plotting things

import cv2

from PIL import Image

import tensorflow as tf 

import random

from keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from keras import optimizers

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
sns.set(style="darkgrid")
heart_disease=pd.read_csv("../input/heart-disease-cleveland-uci/heart_cleveland_upload.csv")
heart_dis=pd.read_csv("../input/heart-disease-cleveland-uci/heart_cleveland_upload.csv")
dummmy_df=heart_dis
dummmy_df['condition'] = dummmy_df['condition'].map({0: 'Normal', 1: 'Heart Attack'})

dummmy_df['sex']=dummmy_df['sex'].map({0:'Female',1:'Male'})

dummmy_df['cp']=dummmy_df['cp'].map({0:'typical angina',1:'atypical angina',2:'non-anginal pain',3:'asymptomatic'})

dummmy_df['restecg']=dummmy_df['restecg'].map({0:'Normal',1:'ST-T abnormal',2:'Left ventricular'})

dummmy_df['exang']=dummmy_df['exang'].map({0:'no',1:'yes'})

dummmy_df['thal']=dummmy_df['thal'].map({0:'Normal',1:'Fixed defect',2:'reversable defect'})
heart_disease.columns
heart_disease.info()
countNoDisease = len(heart_dis[heart_disease.condition == 0])

countHaveDisease = len(heart_dis[heart_disease.condition == 1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(heart_disease.condition))*100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(heart_disease.condition))*100)))
heart_disease.condition.value_counts()
plt.rcParams['figure.figsize'] = (10, 9)
sns.countplot(x="condition", data=heart_dis,hue="condition")

plt.show()
sns.countplot(x="sex", data=heart_dis,hue="condition")

plt.show()
sns.countplot(x="cp", data=heart_dis,hue="condition")

plt.show()
sns.countplot(x="sex", data=heart_dis,hue="condition")

plt.show()
sns.countplot(x="thal", data=heart_dis,hue="condition")

plt.show()
sns.relplot(x='condition',y='age',data=heart_disease,hue='age')
sns.relplot(x="age", y="chol", data=heart_disease,hue='condition',style="condition")
sns.relplot(x="age", y="trestbps", data=heart_disease,hue='condition',style="condition")
sns.countplot(x="fbs", data=heart_dis,hue="condition")

plt.show()

#1= greater than 120 fasting blood sugar level 0=lesser
heart_disease.fbs.value_counts()
sns.countplot(x="restecg", data=heart_dis,hue="condition")

plt.show()
sns.countplot(x="exang", data=heart_dis,hue="condition")

plt.show()
sns.countplot(x="thal", data=heart_dis,hue="condition")

plt.show()

sns.countplot(x="ca", data=heart_dis,hue="condition")

plt.show()
sns.violinplot(heart_dis['condition'], heart_dis['chol'],hue=heart_dis['condition'])

plt.title('Relation of Cholestrol with Condition', fontsize = 20, fontweight = 30)

plt.show()
sns.boxplot(x="condition", y="chol", data=heart_disease)
sns.pairplot(heart_disease)
i=heart_disease[heart_disease['chol']>500].index

heart_disease= heart_disease.drop(i)



#removing outliers
heart_disease
sns.distplot(heart_dis.chol)
sns.countplot(x="cp", data=heart_dis,hue="condition")

plt.show()
sns.boxplot(x="condition", y="thalach", data=heart_disease)
sns.relplot(x='condition',y='thalach',data=heart_disease,hue='age')
f,ax = plt.subplots(figsize=(15, 10))

sns.heatmap(heart_disease.corr(), annot=True, linewidths=0.5, linecolor="red", fmt= '.2f',ax=ax)

plt.show()
sns.countplot(x="slope", data=heart_dis,hue="condition")

plt.show()
Y= heart_disease['condition']

X=heart_disease.drop(['condition','age','fbs','oldpeak'],axis=1)
X.shape
X


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


print("Shape of x_train :", x_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_test :", y_test.shape)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000).fit(x_train,y_train)

log_reg
pred_y=log_reg.predict(x_test)
log_reg.score(x_test,y_test)
CROSS_final = cross_val_score(log_reg, x_train, y_train, cv = 10).mean()

CROSS_final