# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib. pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gender_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

titanic_df = pd.read_csv('/kaggle/input/titanic/train.csv')
gender_df.head()
gender_df.describe()
titanic_df.head(10)
titanic_df.info()
titanic_df.describe()
titanic_df.isnull().sum()
titanic_df.shape
plt.rcParams['figure.figsize'] = (8, 6)

sns.countplot(titanic_df['Embarked'])
print("Survived Passengers is only ",titanic_df['Survived'].value_counts()[1], ' passengers')

plt.rcParams['figure.figsize'] = (8, 6)

sns.countplot(titanic_df['Survived'])

plt.title("No. of Survivors")

plt.show()
female_survived = titanic_df[titanic_df['Sex'] == 'female']['Survived'].sum()/titanic_df[titanic_df['Sex'] == 'female']['Sex'].value_counts()

male_survived = titanic_df[titanic_df['Sex'] == 'male']['Survived'].sum()/titanic_df[titanic_df['Sex'] == 'male']['Sex'].value_counts()

print("Percentage of Male survivors: ","{:.2f}".format(male_survived[0]*100))

print("That means out of ",titanic_df['Sex'].value_counts()[0]," Males only ",titanic_df[titanic_df['Survived']==1]['Sex'].value_counts()[1], " Males Survived")



print("\n")

print("Percentage of Female survivors: ","{:.2f}".format(female_survived[0]*100))

print("That means out of ",titanic_df['Sex'].value_counts()[1]," Females only ",titanic_df[titanic_df['Survived']==1]['Sex'].value_counts()[0], " Females Survived")



plt.rcParams['figure.figsize'] = (8, 6)

sns.barplot(x='Sex',y='Survived',data=titanic_df)

plt.title("Number of Surviver with respect to Sex")

plt.ylabel("Surviver Percentage")

plt.ylim()

plt.show()
df = titanic_df[titanic_df['Survived']==1]

print("Female Survived: ",df['Sex'].value_counts()[0], "\nMale Survived: ",df['Sex'].value_counts()[1])

plt.rcParams['figure.figsize'] = (8, 6)

sns.countplot(df['Sex'])

plt.title("No. of Survivors w.r.t Male and Female")

plt.show()
first_class = titanic_df[titanic_df['Pclass'] == 1]['Survived'].sum()/titanic_df[titanic_df['Pclass'] == 1]['Pclass'].sum()

second_class = titanic_df[titanic_df['Pclass'] == 2]['Survived'].sum()/titanic_df[titanic_df['Pclass'] == 2]['Pclass'].value_counts()

third_class = titanic_df[titanic_df['Pclass'] == 3]['Survived'].sum()/titanic_df[titanic_df['Pclass'] == 3]['Pclass'].value_counts()
print("Percentage of passengers survived from First Class: ","{:.2f}".format(first_class*100))

print("That means out of ",titanic_df[titanic_df['Pclass'] == 1]['Pclass'].sum()," First Class passengers only ",titanic_df[titanic_df['Pclass'] == 1]['Survived'].sum()," Survived")

print("Out ",titanic_df[titanic_df['Pclass'] == 1]['Survived'].sum(),": ",df[df['Pclass']==1]['Sex'].value_counts()[0]," where female and ",df[df['Pclass']==1]['Sex'].value_counts()[1], "male")



print("\nPercentage of passengers survived from Second Class: ","{:.2f}".format(second_class[2]*100))

print("That means out of ",titanic_df[titanic_df['Pclass'] == 2]['Pclass'].value_counts()[2]," Second Class passengers only ",titanic_df[titanic_df['Pclass'] == 2]['Survived'].sum()," Survived")

print("Out ",titanic_df[titanic_df['Pclass'] == 2]['Survived'].sum(),": ",df[df['Pclass']==2]['Sex'].value_counts()[0]," where female and ",df[df['Pclass']==2]['Sex'].value_counts()[1], "male")



print("\nPercentage of passengers survived from Third Class: ","{:.2f}".format(third_class[3]*100))

print("That means out of ",titanic_df[titanic_df['Pclass'] == 3]['Pclass'].value_counts()[3]," Third Class passengers only ",titanic_df[titanic_df['Pclass'] == 3]['Survived'].sum()," Survived")

print("Out ",titanic_df[titanic_df['Pclass'] == 3]['Survived'].sum(),": ",df[df['Pclass']==3]['Sex'].value_counts()[0]," where female and ",df[df['Pclass']==3]['Sex'].value_counts()[1], "male")





plt.rcParams['figure.figsize'] = (8, 6)

sns.barplot(x=titanic_df['Pclass'],y=titanic_df['Survived'],hue=titanic_df['Sex'])



plt.show()
plt.rcParams['figure.figsize'] = (8, 6)

sns.barplot(x='Embarked',y='Survived',data=titanic_df)

plt.show()
copy = titanic_df['Age']

copy.dropna(inplace=True)

print("Check Out the Distribution of Age")

sns.distplot(copy)

plt.show()
print("Mean of Age: ",titanic_df['Age'].mean()),print("Median of Age: ",titanic_df['Age'].median())
titanic_df["Age"].fillna(titanic_df["Age"].median(),inplace=True)

titanic_df["Embarked"].fillna("S", inplace = True)

survived_ages = titanic_df[titanic_df.Survived == 1]["Age"].dropna()

not_survived_ages = titanic_df[titanic_df.Survived == 0]["Age"].dropna()

plt.subplot(1, 2, 1)

sns.distplot(survived_ages, kde=False)

plt.axis([0, 100, 0, 100])

plt.title("Survived")

plt.ylabel("Proportion")

plt.subplot(1, 2, 2)

sns.distplot(not_survived_ages, kde=False)

plt.axis([0, 100, 0, 100])

plt.title("Didn't Survive")

plt.subplots_adjust(right=1.7)

plt.show()
sns.pairplot(titanic_df)

plt.show()