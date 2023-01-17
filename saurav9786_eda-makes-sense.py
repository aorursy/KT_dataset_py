

import numpy as np 

import pandas as pd 

import seaborn as sns

sns.set(color_codes=True)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the Dataset

pima_data=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

#Top 5 Dataset

pima_data.head()

#Display the last 5 dataset



pima_data.tail()

#Shape of the data 



pima_data.shape
#Info of the data



pima_data.info()

#Five Point summary of the data 



pima_data.describe()

import pandas_profiling as pp

pp.ProfileReport(pima_data)

# To check the missing values in the dataset



pima_data.isnull().values.any()
#Replace 0 to NaN



d=pima_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=pima_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

d.head()


# Find the number of Missing values



d.isnull().sum()[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]
#Replace NaN to mean value to explore dataset



pima_data['Glucose'].fillna(pima_data['Glucose'].median(),inplace=True)

pima_data['BloodPressure'].fillna(pima_data['BloodPressure'].median(),inplace=True)

pima_data['SkinThickness'].fillna(pima_data['SkinThickness'].median(),inplace=True)

pima_data['Insulin'].fillna(pima_data['Insulin'].median(),inplace=True)

pima_data['BMI'].fillna(pima_data['BMI'].median(),inplace=True)

pima_data.head()
# Analysing the Outcome



# To get the number of diabetic and Healthy person



pima_data.groupby('Outcome').size()
# countplot----Plot the frequency of the Outcome



fig1, ax1 = plt.subplots(1,2,figsize=(8,8))



#It shows the count of observations in each categorical bin using bars



sns.countplot(pima_data['Outcome'],ax=ax1[0])



#Find the % of diabetic and Healthy person



labels = 'Diabetic', 'Healthy'



pima_data.Outcome.value_counts().plot.pie(labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
# Histogram 



pima_data.hist(figsize=(15,10))


# Distplot



fig, ax2 = plt.subplots(4, 2, figsize=(16, 16))

sns.distplot(pima_data['Pregnancies'],ax=ax2[0][0])

sns.distplot(pima_data['Glucose'],ax=ax2[0][1])

sns.distplot(pima_data['BloodPressure'],ax=ax2[1][0])

sns.distplot(pima_data['SkinThickness'],ax=ax2[1][1])

sns.distplot(pima_data['Insulin'],ax=ax2[2][0])

sns.distplot(pima_data['BMI'],ax=ax2[2][1])

sns.distplot(pima_data['DiabetesPedigreeFunction'],ax=ax2[3][0])

sns.distplot(pima_data['Age'],ax=ax2[3][1])
# boxplot



fig=plt.figure(figsize=(20,3))



for i in np.arange(1,7):

    data3=plt.subplot(1,7,i,title=pima_data.columns[i])

    sns.boxplot(pima_data[pima_data.columns[i]])
# pairplot--Multiple relationship of scatterplot



sns.pairplot(pima_data,hue='Outcome')
# corrlation matrix 



cor=pima_data.corr()

cor
# correlation plot---heatmap



sns.heatmap(cor,annot=True)
a=pd.Series([])

for i in pima_data.index:

    if(pima_data.loc[i:i,]['Age']<=24).bool():

        a=a.append(pd.Series(['21-24']))

    elif(pima_data.loc[i:i,]['Age']<=30).bool():

        a=a.append(pd.Series(['25-30']))

    elif(pima_data.loc[i:i,]['Age']<=40).bool():

        a=a.append(pd.Series(['31-40']))

    elif(pima_data.loc[i:i,]['Age']<=55).bool():

        a=a.append(pd.Series(['41-55']))

    else:

        a=a.append(pd.Series(['>55']))

a.reset_index(drop=True,inplace=True)

pima_data['Age']=a

pima_data.head()



#Find the number of diabetic person in each age group



data1=pima_data[pima_data['Outcome']==1].groupby('Age')[['Outcome']].count()

data1

data1.head()
# Percentage of diabetic Person in each age group



data2=pima_data.groupby('Age')[['Outcome']].count()

data1['Diabetic %']=(data1['Outcome']/data2['Outcome'])*100

data1
#4.1 barplot



sns.barplot(data1.index,data1['Diabetic %'])
#6.Crosstab gives the fregency table information ----Pregnancies



pd.crosstab(pima_data['Pregnancies'],pima_data['Outcome'])
# Categorical vs Continuous ----Outcome vs Pregnancies



fig, ax2 = plt.subplots(3, 2, figsize=(12, 8))

sns.boxplot(x="Outcome", y="Pregnancies", data=pima_data,ax=ax2[0][0])

sns.barplot(pima_data['Outcome'], pima_data['Pregnancies'],ax=ax2[0][1])

sns.stripplot(pima_data['Outcome'], pima_data['Pregnancies'], jitter=True,ax=ax2[1][0])

sns.swarmplot(pima_data['Outcome'], pima_data['Pregnancies'], ax=ax2[1][1])

sns.violinplot(pima_data['Outcome'], pima_data['Pregnancies'], ax=ax2[2][0])

sns.countplot(x='Pregnancies',hue='Outcome',data=pima_data,ax=ax2[2][1])
# Categorical vs Continuous ---- Outcome vs Glucose 



fig, ax2 = plt.subplots(2, 2, figsize=(12, 8))

sns.boxplot(x="Outcome", y="Glucose", data=pima_data,ax=ax2[0][0])

sns.barplot(pima_data['Outcome'], pima_data['Glucose'],ax=ax2[0][1])

sns.stripplot(pima_data['Outcome'], pima_data['Glucose'], jitter=True,ax=ax2[1][0])

sns.swarmplot(pima_data['Outcome'], pima_data['Glucose'], ax=ax2[1][1])
# Categorical vs Continuous ---- Outcome vs BloodPressure



fig, ax2 = plt.subplots(2, 2, figsize=(12, 8))

sns.boxplot(x="Outcome", y="BloodPressure", data=pima_data,ax=ax2[0][0])

sns.barplot(pima_data['Outcome'], pima_data['BloodPressure'],ax=ax2[0][1])

sns.stripplot(pima_data['Outcome'], pima_data['BloodPressure'], jitter=True,ax=ax2[1][0])

sns.swarmplot(pima_data['Outcome'], pima_data['BloodPressure'], ax=ax2[1][1])
# Categorical vs Continuous ----Outcome vs SkinThickness  



fig, ax2 = plt.subplots(2, 2, figsize=(12, 8))

sns.boxplot(x="Outcome", y="SkinThickness", data=pima_data,ax=ax2[0][0])

sns.barplot(pima_data['Outcome'], pima_data['SkinThickness'],ax=ax2[0][1])

sns.stripplot(pima_data['Outcome'], pima_data['SkinThickness'], jitter=True,ax=ax2[1][0])

sns.swarmplot(pima_data['Outcome'], pima_data['SkinThickness'], ax=ax2[1][1])
# Categorical vs Continuous ----Outcome vs Insulin  



fig, ax2 = plt.subplots(2, 2, figsize=(12, 8))

sns.boxplot(x="Outcome", y="Insulin", data=pima_data,ax=ax2[0][0])

sns.barplot(pima_data['Outcome'], pima_data['Insulin'],ax=ax2[0][1])

sns.stripplot(pima_data['Outcome'], pima_data['Insulin'], jitter=True,ax=ax2[1][0])

sns.swarmplot(pima_data['Outcome'], pima_data['Insulin'], ax=ax2[1][1])
# Categorical vs Continuous ----Outcome vs BMI



fig, ax2 = plt.subplots(2, 2, figsize=(12, 8))

sns.boxplot(x="Outcome", y="BMI", data=pima_data,ax=ax2[0][0])

sns.barplot(pima_data['Outcome'], pima_data['BMI'],ax=ax2[0][1])

sns.stripplot(pima_data['Outcome'], pima_data['BMI'], jitter=True,ax=ax2[1][0])

sns.swarmplot(pima_data['Outcome'], pima_data['BMI'], ax=ax2[1][1])
# Categorical vs Continuous ----Outcome vs DiabetesPedigreeFunction



fig, ax2 = plt.subplots(2, 2, figsize=(12, 8))

sns.boxplot(x="Outcome", y="DiabetesPedigreeFunction", data=pima_data,ax=ax2[0][0])

sns.barplot(pima_data['Outcome'], pima_data['DiabetesPedigreeFunction'],ax=ax2[0][1])

sns.stripplot(pima_data['Outcome'], pima_data['DiabetesPedigreeFunction'], jitter=True,ax=ax2[1][0])

sns.swarmplot(pima_data['Outcome'], pima_data['DiabetesPedigreeFunction'], ax=ax2[1][1])
# lmplot---linear Regression plots

pima_data=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')



sns.pointplot(pima_data['Pregnancies'], pima_data['Age'], hue=pima_data['Outcome'])

sns.jointplot(pima_data['Pregnancies'], pima_data['Age'], kind='hex')

sns.lmplot(x='Pregnancies',y='Age',data=pima_data,hue='Outcome')
sns.pointplot(pima_data['Insulin'], pima_data['SkinThickness'], hue=pima_data['Outcome'])

sns.jointplot(pima_data['Insulin'], pima_data['SkinThickness'], kind='hex')

sns.lmplot(x='Insulin',y='SkinThickness',data=pima_data,hue='Outcome')
sns.pointplot(pima_data['BMI'], pima_data['SkinThickness'], hue=pima_data['Outcome'])

sns.jointplot(pima_data['BMI'], pima_data['SkinThickness'], kind='hex')

sns.lmplot(x='BMI',y='SkinThickness',data=pima_data,hue='Outcome')
sns.pointplot(pima_data['Insulin'], pima_data['Glucose'], hue=pima_data['Outcome'])

sns.jointplot(pima_data['Insulin'], pima_data['Glucose'], kind='hex')

sns.lmplot(x='Insulin',y='Glucose',data=pima_data,hue='Outcome')