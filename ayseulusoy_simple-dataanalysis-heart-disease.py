import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sea

import warnings 

import os
data=pd.read_csv("../input/heart.csv")

print(type(data))

data.isnull().sum() #we are checking the data and it is showing us to Null values on the data
data.shape #it is showing us counts of rows and columns
data.columns #it is showing us name of columns
data.head() #it is showing us first 5 datas
data.head(30) #it is showing us first 30 datas
data.tail() #it is showing us last 5 datas
data.tail(50) #it is showing us last 50 datas
data.describe() #it is showing us data's count,mean,standart deviation...
data.mean() #it is showing us data's mean
data.std() #it is showing us data's standart deviation
data.iloc[0:20,:2].plot()
plt.figure(figsize=(15,15))

sea.countplot(x=data.age,hue=data.sex)

plt.show()



data_male=len(data[data['sex']==1])

data_female=len(data[data['sex']==0])
plt.figure(figsize=(10,10))

labels='Male','Female'

sizes=[data_male,data_female]

colors=['orange','blue']

explode=(0,0.01)





plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=False,startangle=90)

plt.axis('equal')

plt.show()
chest_pain_ha=[]

for cp in data['cp'].unique():

    chest_pain_ha.append(len(data[data['cp']==cp])) #we are finding the counts for the different chest pain numbers

plt.figure(figsize=(10,10))

sea.barplot(x=data['cp'].unique(),y=chest_pain_ha)

plt.show()
rbpc=[] #resting blood pressure count

for bp in data['trestbps'].unique():

    rbpc.append(len(data[data['trestbps']==bp]))

rbpc=pd.DataFrame(rbpc,columns=['rbpc']) #making columns

rbp=pd.DataFrame(data['trestbps'].unique(),columns=['rbp'])



resting_blood_count=pd.concat([rbpc,rbp],axis=1) #combine the columns

resting_blood_count=resting_blood_count.sort_values(by='rbpc',ascending=False) #sorting the data by 'rbpc'



plt.figure(figsize=(20,20))

sea.barplot(x=resting_blood_count.rbp,y=resting_blood_count.rbpc)

plt.show()
plt.figure(figsize=(20,20))

sea.jointplot(y=data.age,x=data['trestbps'],kind='kde') #we finding rbp by the age

plt.show()
data_40_50=data[(data['age']>=40)&(data['age']<50)]

data_50_60=data[(data['age']>=50)&(data['age']<60)]

data_60_70=data[(data['age']>=60)&(data['age']<70)]



plt.figure(figsize=(20,20))

sea.jointplot(y=data.chol,x=data['trestbps'],kind='kde')

plt.show()   #we are drawing a figure for everybody by the cholesterol and resting



plt.figure(figsize=(20,20))

sea.jointplot(y=data_40_50.chol,x=data_40_50['trestbps'],kind='kde')

plt.show() #we are drawing a figure for 40/50 year olds people by the cholesterol and resting



plt.figure(figsize=(20,20))

sea.jointplot(y=data_50_60.chol,x=data_50_60['trestbps'],kind='kde')

plt.show() #we are drawing a figure for 50/60 year olds people by the cholesterol and resting



plt.figure(figsize=(20,20))

sea.jointplot(y=data_60_70.chol,x=data_60_70['trestbps'],kind='kde')

plt.show() #we are drawing a figure for 60/70 year olds people by the cholesterol and resting

data_angina=data[data['exang']==1]

data_not_angina=data[data['exang']==0]

data_angina_age=[]

data_not_angina_age=[]

for age in data_angina['age'].unique():

    data_angina_age.append(len(data[data['age']==age])) #We are finding count of who is feeling angina pain for every age

for age in data_not_angina['age'].unique():

    data_not_angina_age.append(len(data[data['age']==age])) #We are finding count of who is not feeling angina pain for every age



plt.figure(figsize=(10,10))

sea.countplot(data.exang)

plt.show() #we are drawing a figure for how many person feel angina pain and not feel angina pain



plt.figure(figsize=(10,10))

sea.barplot(x=data_angina.age.unique(),y=data_angina_age)

plt.show()

plt.figure(figsize=(10,10))

sea.barplot(x=data_not_angina.age.unique(),y=data_not_angina_age)

plt.show()