import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm 

import cufflinks as cf

%matplotlib inline

sns.set_style('whitegrid')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df
df.info()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='DEATH_EVENT',data=df,palette='RdBu_r')
sns.countplot(x='DEATH_EVENT',data=df,hue='anaemia',palette='RdBu_r')
sns.countplot(x='DEATH_EVENT',data=df,hue='diabetes',palette='RdBu_r')
sns.countplot(x='DEATH_EVENT',data=df,hue='high_blood_pressure',palette='RdBu_r')
sns.countplot(x='DEATH_EVENT',data=df,hue='smoking',palette='RdBu_r')
sns.countplot(x='DEATH_EVENT',data=df,hue='sex',palette='RdBu_r')
cf.go_offline()

df['age'].iplot(kind='hist',bins=30,color='red')
df['creatinine_phosphokinase'].iplot(kind='hist',bins=30,color='green')
df['ejection_fraction'].iplot(kind='hist',bins=30,color='yellow')
df['platelets'].iplot(kind='hist',bins=30,color='magenta')
df['time'].iplot(kind='hist',bins=30,color='orange')
df['serum_creatinine'].iplot(kind='hist',bins=30,color='blue')
df['serum_sodium'].iplot(kind='hist',bins=30,color='pink')
df_data = pd.get_dummies(df,columns=['anaemia','diabetes','high_blood_pressure','sex','smoking'],drop_first=True)

df_data
df_data.dropna(inplace=True)

df_data.info()
x = df_data[['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time','anaemia_1','diabetes_1','high_blood_pressure_1','sex_1','smoking_1']]

y = df_data['DEATH_EVENT']
predictions = logmodel.predict(x)
probability = logmodel.predict_proba(x)
sns.regplot(x='age', y='DEATH_EVENT', data=df_data, logistic=True)
sns.regplot(x='creatinine_phosphokinase', y='DEATH_EVENT', data=df_data, logistic=True)
sns.regplot(x='ejection_fraction', y='DEATH_EVENT', data=df_data, logistic=True)
sns.regplot(x='platelets', y='DEATH_EVENT', data=df_data, logistic=True)
sns.regplot(x='serum_creatinine', y='DEATH_EVENT', data=df_data, logistic=True)
sns.regplot(x='serum_sodium', y='DEATH_EVENT', data=df_data, logistic=True)
df = {

    'age' : df['age'],

    'DEATH_EVENT' : predictions

}



df = pd.DataFrame(df)



df.to_csv('Output.csv', index = False)



df