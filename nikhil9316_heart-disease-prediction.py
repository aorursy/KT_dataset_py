import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
Data_heart=pd.read_csv(r'../input/heart-disease-uci/heart.csv')
Data_heart_copy=Data_heart.copy()
Data_heart_copy.head()
msno.bar(Data_heart)
plt.show()
visuals={"sex":{1:"Male",0:"Female"},
         "cp":{0:"typical angina",1: "atypical angina" ,2: "non-anginal pain" ,3: "asymptomatic"},
         "fbs":{0:"<=120",1:">120"},
         "exang":{0:"no",1:"yes"},
         "restecg" :{0:"normal" ,1:"ST-T wave abnormality",2:"probable or definite left ventricular hypertrophy"},
         "target" :{ 0:"No Heart Disease",1 : "heart-disease"},
         "slope" :{2 : "upsloping",1 :"flat",0 : "downsloping"},
         "thal" :{ 1 : "fixed defect",0 : "normal",2 : "reversable defect",3:"NA"}
         
}
Data_heart_copy.replace(visuals,inplace=True)
plt.figure(figsize=(15,15))
for i, col in enumerate(['age', 'trestbps', 'chol','thalach','oldpeak', 'ca']):
    plt.subplot(3,2,i+1)
    sns.distplot(Data_heart_copy[col],hist=False)
plt.show()
plt.figure(figsize=(25,35))
for i, col in enumerate(['sex', 'cp', 'fbs', 'restecg','exang','slope', 'thal', 'target']):
    plt.subplot(4,2,i+1)
    sns.countplot(x=col,data=Data_heart_copy)
plt.show()
sns.catplot(x='target',y='age',hue='sex',data=Data_heart_copy,kind='violin')
plt.show()
sns.catplot('target',col='sex',data=Data_heart_copy,kind='count')
plt.show()
sns.catplot('target',col='cp',data=Data_heart_copy,kind='count')
plt.show()
sns.catplot('target',col='restecg',data=Data_heart_copy,kind='count')
plt.show()
sns.catplot('target',col='thal',data=Data_heart_copy,kind='count')
plt.show()
sns.boxplot(x="target",y="thalach",data=Data_heart_copy)
plt.ylabel("Max. heart rate achieved during thalium stress test ")
plt.xlabel("Disease Condition")
plt.show()
plt.show()
sns.boxplot(x="target",y="chol",data=Data_heart_copy)
plt.ylabel("Cholestrol")
plt.xlabel("Disease Condition")
plt.show()
sns.boxplot(x="target",y="trestbps",data=Data_heart_copy)
plt.ylabel("Resting Blood Pressure")
plt.xlabel("Disease Condition")
plt.show()