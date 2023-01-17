# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
diab = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

print(diab)

print(type(diab))
diab.info()
diab.isnull().sum()
column = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']



diab[column] = diab[column].replace(0, np.nan)

print(diab)
diab.isnull().sum()
diab = diab.dropna()

print(diab)
import seaborn as sns; sns.set(style="ticks")

g = sns.pairplot(diab, hue="Outcome")
f, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=False)

feature_name = "BMI"

sns.kdeplot(diab[feature_name], color='black', shade=True, ax=axes[0, 0], label=feature_name)

sns.kdeplot(diab[diab.Outcome==1][feature_name], color='r', shade=True, ax=axes[0, 1], label="diabetics")

sns.kdeplot(diab[diab.Outcome==0][feature_name], color='g', shade=True, ax=axes[0, 1], label="non-diabete")

sns.boxplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 0])

sns.violinplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 1])
f, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=False)

feature_name = "Pregnancies"

sns.kdeplot(diab[feature_name], color='black', shade=True, ax=axes[0, 0], label=feature_name)

sns.kdeplot(diab[diab.Outcome==1][feature_name], color='r', shade=True, ax=axes[0, 1], label="diabetics")

sns.kdeplot(diab[diab.Outcome==0][feature_name], color='g', shade=True, ax=axes[0, 1], label="non-diabete")

sns.boxplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 0])

sns.violinplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 1])
f, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=False)

feature_name = "Glucose"

sns.kdeplot(diab[feature_name], color='black', shade=True, ax=axes[0, 0], label=feature_name)

sns.kdeplot(diab[diab.Outcome==1][feature_name], color='r', shade=True, ax=axes[0, 1], label="diabetics")

sns.kdeplot(diab[diab.Outcome==0][feature_name], color='g', shade=True, ax=axes[0, 1], label="non-diabete")

sns.boxplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 0])

sns.violinplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 1])
f, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=False)

feature_name = "BloodPressure"

sns.kdeplot(diab[feature_name], color='black', shade=True, ax=axes[0, 0], label=feature_name)

sns.kdeplot(diab[diab.Outcome==1][feature_name], color='r', shade=True, ax=axes[0, 1], label="diabetics")

sns.kdeplot(diab[diab.Outcome==0][feature_name], color='g', shade=True, ax=axes[0, 1], label="non-diabete")

sns.boxplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 0])

sns.violinplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 1])
f, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=False)

feature_name = "SkinThickness"

sns.kdeplot(diab[feature_name], color='black', shade=True, ax=axes[0, 0], label=feature_name)

sns.kdeplot(diab[diab.Outcome==1][feature_name], color='r', shade=True, ax=axes[0, 1], label="diabetics")

sns.kdeplot(diab[diab.Outcome==0][feature_name], color='g', shade=True, ax=axes[0, 1], label="non-diabete")

sns.boxplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 0])

sns.violinplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 1])
f, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=False)

feature_name = "Insulin"

sns.kdeplot(diab[feature_name], color='black', shade=True, ax=axes[0, 0], label=feature_name)

sns.kdeplot(diab[diab.Outcome==1][feature_name], color='r', shade=True, ax=axes[0, 1], label="diabetics")

sns.kdeplot(diab[diab.Outcome==0][feature_name], color='g', shade=True, ax=axes[0, 1], label="non-diabete")

sns.boxplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 0])

sns.violinplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 1])
f, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=False)

feature_name = "DiabetesPedigreeFunction"

sns.kdeplot(diab[feature_name], color='black', shade=True, ax=axes[0, 0], label=feature_name)

sns.kdeplot(diab[diab.Outcome==1][feature_name], color='r', shade=True, ax=axes[0, 1], label="diabetics")

sns.kdeplot(diab[diab.Outcome==0][feature_name], color='g', shade=True, ax=axes[0, 1], label="non-diabete")

sns.boxplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 0])

sns.violinplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 1])
f, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=False)

feature_name = "Age"

sns.kdeplot(diab[feature_name], color='black', shade=True, ax=axes[0, 0], label=feature_name)

sns.kdeplot(diab[diab.Outcome==1][feature_name], color='r', shade=True, ax=axes[0, 1], label="diabetics")

sns.kdeplot(diab[diab.Outcome==0][feature_name], color='g', shade=True, ax=axes[0, 1], label="non-diabete")

sns.boxplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 0])

sns.violinplot(data=diab, x='Outcome', y=feature_name, ax=axes[1, 1])
sns.pairplot(data=diab, vars=['SkinThickness', 'BMI'], hue='Outcome')
plt.figure(figsize=[9,9])

sns.regplot(data=diab, x='BMI', y='SkinThickness', scatter_kws={'color':'b', 'alpha':0.3}, line_kws={'color':'r', 'alpha':0.7})
sns.lmplot(data=diab, x='BMI', y='SkinThickness', hue='Outcome')
sns.pairplot(data=diab, vars=['Glucose', 'BloodPressure'], hue='Outcome')
outcome1 = diab.query("Outcome==1")

outcome0 = diab.query("Outcome==0")

f, ax = plt.subplots(figsize=(9,16))

ax.set_aspect('equal')

ax=sns.kdeplot(outcome1['Glucose'], outcome1['BloodPressure'], cmap='Reds', shade=True, shade_lowst=False, alpha=0.6)

ax=sns.kdeplot(outcome0['Glucose'], outcome0['BloodPressure'], cmap='Blues', shade=True, shade_lowst=False, alpha=0.6)
piex = [outcome0['Outcome'].count(), outcome1['Outcome'].count()]

explode=[0.1, 0]

plt.figure(figsize=[9,9])

plt.pie(x=piex, labels=['non-diabetics', 'diabetics'], shadow=True, autopct='%.1f%%', textprops={'fontsize':20}, explode=explode, pctdistance=0.5)
from collections import Counter

from imblearn.combine import SMOTEENN
diab1=diab.copy()

diab1.drop(columns='Outcome')

print(diab1)
sm=SMOTEENN()

classification=diab['Outcome']

train_set=diab.drop(columns='Outcome')

train_set, classification=sm.fit_resample(train_set, classification)

Counter(classification)
train_set.info()
piex=Counter(classification)

piexx=[piex[0], piex[1]]

explode=[0.1, 0]

plt.figure(figsize=[9,9])

plt.pie(x=piexx, labels=['non-diabetics', 'diabetics'], shadow=True, autopct='%.1f%%', textprops={'fontsize':20}, explode=explode, pctdistance=0.5)
from sklearn import preprocessing

import pandas as pd

#diabppp=diab.drop(['Outcome'], axis=1)

#print(type(preprocessing.scale(diabppp)))

#print(preprocessing.scale(diabppp))
plt.figure(figsize=(16,9))

plt.subplot(211)

diabp=diab.drop(['Outcome'], axis=1)

for s in diabp.columns:

    sns.kdeplot(diabp[s], shade=True, label=s)



plt.subplot(212)

scale = pd.DataFrame(preprocessing.scale(diabp))

scale.columns = diabp.columns

for s in scale.columns:

    sns.kdeplot(scale[s], shade=True, label=s)