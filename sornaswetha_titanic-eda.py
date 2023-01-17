# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-pytho



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats  as stats
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.shape
train.describe()
train.info()
# plot a pie and bar plot using subplots(1,2,figsize=(8,5))

fig,axes = plt.subplots(1,2,figsize=(8,5))

train["Sex"].value_counts().plot(kind="bar", ax=axes[0], color =['DarkRed','indianred'])

train["Sex"].value_counts().plot(kind="pie",ax=axes[1],autopct='%0.2f' ,colormap="Reds")

plt.show()
# plot a pie and bar plot using subplots(1,2,figsize=(8,5))

fig,axes = plt.subplots(1,2,figsize=(8,5))

train["Embarked"].value_counts().plot(kind="bar", ax=axes[0],color =['DarkRed','indianred','darksalmon'])

train["Embarked"].value_counts().plot(kind="pie",ax=axes[1],autopct='%0.2f', colormap="Reds")

plt.show()

# plot a pie and bar plot using subplots(1,2,figsize=(8,5))

fig,axes = plt.subplots(1,2,figsize=(8,5))

train["Pclass"].value_counts().plot(kind="bar", ax=axes[0],color =['DarkRed','indianred','darksalmon'])

train["Pclass"].value_counts().plot(kind="pie",ax=axes[1],autopct='%0.2f',colormap="Reds")

plt.show()

sns.kdeplot( data= train["Age"] ,color = "darksalmon")

plt.show()
sns.countplot(x="Sex", hue="Survived", data=train, palette="Reds" ,)

plt.show()



#print percentages of females vs. males that survive

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
sns.countplot(x="Pclass", hue="Survived", data=train, palette="Reds" ,)

plt.show()

#print percentages of 1st vs. 2nd and 3rd class

print("Percentage of 1st class who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of 2nd class who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of 3rd class who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
# create subplot plot

fig,axes=plt.subplots(1,2,figsize=(8,4))

# create violinplot plot using groupby

sns.violinplot (x="Pclass",y="Age",hue="Survived", data=train,split=True,ax=axes[0], palette="Reds" )

sns.violinplot (x="Sex",y="Age",hue="Survived", data=train,split=True,ax=axes[1] , palette="Reds" )

plt.show()
# create subplot plot

fig,axes=plt.subplots(2,2,figsize=(8,8))

# create Bar (count) plot for Embarked vs. No. Of Passengers Boarded

sns.countplot(x="Embarked",data=train,ax=axes[0][0],palette="Reds")

# create Bar (count) plot for Embarked vs. Male-Female Split

sns.countplot(x="Embarked",hue="Sex",data=train,ax=axes[0][1],palette="Reds")

# create Bar (count) plot for Embarked vs Survived

sns.countplot(x="Embarked",hue="Survived",data=train,ax=axes[1][0],palette="Reds")

# create Bar (count) plot for Embarked vs Pclass

sns.countplot(x="Embarked",hue="Pclass",data=train,ax=axes[1][1],palette="Reds")

plt.show()
ax= sns.boxplot(x="Pclass", y="Age", data=train ,palette = "Reds")

ax= sns.stripplot(x="Pclass", y="Age", data=train, jitter=True, edgecolor="gray" , palette = "Reds")

plt.show()
#create crosstab

tab = pd.crosstab(train['Sex'], train['Survived'])

print(tab)



dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True , color=['indianred','darksalmon'])

dummy = plt.xlabel('Port embarked')

dummy = plt.ylabel('Percentage')
sns.lmplot(x='Age', y='Fare', hue='Survived',  data=train.loc[train['Survived'].isin([1,0])], fit_reg=False ,palette = "Reds")

plt.show()
sns.heatmap(train.corr(),cmap="Reds",annot=True)

plt.show()