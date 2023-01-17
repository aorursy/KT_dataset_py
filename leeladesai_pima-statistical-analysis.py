# Import Required Packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)

print("Packages LOADED")
#Load the dataset as dataframe using pandas lib

#data = pd.read_csv('diabetes.csv')

data = pd.read_csv("../input/diabetes.csv")

data.shape
data.info()
data.head(10)
data.tail(10)
normal = data[data['Outcome'] == 0].count()['Outcome']

diabetic = data[data['Outcome'] == 1].count()['Outcome']

(normal,diabetic)

data.count()
Percentage_Of_Diabetics = {"Normal" : normal/data.count()['Outcome']*100,

                            "Diabetic" : diabetic/data.count()['Outcome']*100}

Percentage_Of_Diabetics
data.describe().T


plt.figure(figsize=(20, 20))



for column_index, column in enumerate(data.columns):

    if column == 'Outcome':

        continue

    plt.subplot(4, 2, column_index + 1)

    sns.distplot(data[column],bins = 20)
zeroGlucose = data[data['Glucose'] == 0].count()['Glucose']

zeroBloodPressure = data[data['BloodPressure'] == 0].count()['BloodPressure']

zeroSkinThickness = data[data['SkinThickness'] == 0].count()['SkinThickness']

zeroInsulin = data[data['Insulin'] == 0].count()['Insulin']

zeroBMI = data[data['BMI'] == 0].count()['BMI']

zeros = {"zeroGlucose":zeroGlucose,"zeroBloodPressure":zeroBloodPressure,"zeroSkinThickness":zeroSkinThickness,"zeroInsulin":zeroInsulin,"zeroBMI":zeroBMI}

zeros
(sum(zeros.values())/sum(data.count()))*100
data['Glucose'] = data['Glucose'].replace(

    to_replace=0, value=data['Glucose'].mean())

data['BloodPressure'] = data['BloodPressure'].replace(

    to_replace=0, value=data['BloodPressure'].mean())

data['SkinThickness'] = data['SkinThickness'].replace(

    to_replace=0, value=data['SkinThickness'].mean())

data['Insulin'] = data['Insulin'].replace(

    to_replace=0, value=data['Insulin'].mean())

data['BMI'] = data['BMI'].replace(

    to_replace=0, value=data['BMI'].mean())
zeroGlucose = data[data['Glucose'] == 0].count()['Glucose']

zeroBloodPressure = data[data['BloodPressure'] == 0].count()['BloodPressure']

zeroSkinThickness = data[data['SkinThickness'] == 0].count()['SkinThickness']

zeroInsulin = data[data['Insulin'] == 0].count()['Insulin']

zeroBMI = data[data['BMI'] == 0].count()['BMI']

zeros = {"zeroGlucose":zeroGlucose,"zeroBloodPressure":zeroBloodPressure,"zeroSkinThickness":zeroSkinThickness,"zeroInsulin":zeroInsulin,"zeroBMI":zeroBMI}

zeros
data.describe().T


plt.figure(figsize=(20, 20))



for column_index, column in enumerate(data.columns):

    if column == 'Outcome':

        continue

    plt.subplot(4, 2, column_index + 1)

    sns.distplot(data[column],bins = 20)
data.skew()


plt.figure(figsize=(20, 20))



for column_index, column in enumerate(data.columns):

    if column == 'Outcome':

        continue

    plt.subplot(4, 2, column_index + 1)

    sns.boxplot(data[column])

plt.figure(figsize=(20, 20))



for column_index, column in enumerate(data.columns):

    if column == 'Outcome':

        continue

    plt.subplot(4, 4, column_index + 1)

    sns.violinplot(x='Outcome', y=column, data=data)
sns.pairplot(data, hue = 'Outcome')
#sns.regplot(x='Insulin', y= 'SkinThickness', data=data)
#sns.lmplot(x='Insulin', y= 'Glucose', data=data)
corr = data.corr()

corr
plt.figure(figsize=(20, 20))

#sns.set(font_scale=2)

sns.heatmap(corr, vmax=.8, linewidths=0.05,

            square=True,annot=True,cmap='coolwarm',linecolor="black")