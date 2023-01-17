# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
#Obtain the numerical variables

numerical_var = train.select_dtypes(include=[np.number])

numerical_var.dtypes
#Obtain the categorical variables

categorical_var = train.select_dtypes(exclude=[np.number])

categorical_var.dtypes
#plot Age using pandas

train['Age'].plot(kind='hist', bins=50);

plt.title('Age of Titanic Passengers', fontsize=18);

plt.xlabel('Age', fontsize=16);

plt.ylabel('Frequency', fontsize=16);
#plot Age using seaborn

sns.distplot(train['Age'], kde=False, color='red');

plt.title('Age of Titanic Passengers', fontsize=18);

plt.xlabel('Age', fontsize=16);

plt.ylabel('Frequency', fontsize=16);
#plot multiple histograms with Seaborn

survived = train[train.Survived == 0]

sns.distplot(train['Age'],  kde=False, label='Survived');



not_survived = train[train.Survived == 1]

sns.distplot(not_survived['Age'], kde=False, label='Not Survived');



plt.legend(prop={'size': 12});

plt.title('Age of Titanic Passengers', fontsize=18);

plt.xlabel('Age', fontsize=16);

plt.ylabel('Count', fontsize=16);
#plot Survived using pandas

train['Survived'].value_counts().plot(kind='bar');

plt.title('Survival of Titanic Passengers', fontsize=18);

plt.xlabel('Survived', fontsize=16);

plt.ylabel('Count', fontsize=16);
#plot Survived using Seaborn

sns.countplot(x="Survived", data=train);

plt.title('Survival of Titanic Passengers', fontsize=18);

plt.xlabel('Survived', fontsize=16);

plt.ylabel('Count', fontsize=16);
#We can also plot using the y-axis

sns.countplot(y="Survived", data=train);

plt.title('Survival of Titanic Passengers', fontsize=18);

plt.xlabel('Count', fontsize=16);

plt.ylabel('Survived', fontsize=16);
#plot Age and Fare using pandas

train.plot(x='Age', y='Fare', kind='scatter');

plt.title('Age and Fare of Titanic Passengers', fontsize=18);

plt.xlabel('Age', fontsize=16);

plt.ylabel('Fare', fontsize=16);
#plot Age and Fare using Seaborn

sns.scatterplot(x='Age', y='Fare', data=train);

plt.title('Age and Fare of the Titanic Passengers', fontsize=18);

plt.xlabel('Age', fontsize=16);

plt.ylabel('Fare', fontsize=16);
#plot a heatmap using seaborn

sns.heatmap(train.corr(), cmap="YlGnBu");
#plot Survived and Sex using pandas

survived_and_sex = pd.crosstab(train['Survived'], train['Sex'])

survived_and_sex.plot(kind='bar', stacked=True, color=['red','blue'], grid=False,alpha=0.5);

plt.title('Survival of Titanic Passengers by Sex', fontsize=18);

plt.xlabel('Survived', fontsize=16);
#plot Survived and Sex using Seaborn

sns.countplot(x="Survived", hue='Sex', data=train);

plt.title('Survival of Titanic Passengers', fontsize=18);
#plot Survived and Sex using Seaborn

sns.countplot(y="Survived", hue='Sex', data=train);

plt.title('Survival of Titanic Passengers', fontsize=18);
#boxplot Sex and Fare using pandas

train.boxplot(by='Sex', column=['Fare'], grid=False);
#boxplot Sex and Fare using Seaborn

sns.boxplot(y='Fare', x='Sex', data=train, width=0.5, palette="colorblind");

plt.title('Fare of Titanic Passengers by Sex', fontsize=18);

plt.xlabel('Sex', fontsize=16);

plt.ylabel('Fare', fontsize=16);
#plot multiple histograms with Seaborn

male = train[train.Sex == 'male']

sns.distplot(male['Fare'],  kde=False, label='Male');



female = train[train.Sex == 'female']

sns.distplot(female['Fare'], kde=False, label='Female');



plt.legend(prop={'size': 12});

plt.title('Fare of Titanic Passengers by Sex', fontsize=18);

plt.xlabel('Fare', fontsize=16);

plt.ylabel('Count', fontsize=16);
#Multivariate Analysis in Pandas

ax = male.plot.scatter(x='Age', y='Fare', color='DarkBlue', label='Male');

female.plot.scatter(x='Age', y='Fare', color='DarkGreen', label='Female', ax=ax);

plt.title('Fare and Age of Titanic Passengers by Sex', fontsize=18);

plt.xlabel('Age', fontsize=16);

plt.ylabel('Fare', fontsize=16);
#Multivariate Analysis in Seaborn

sns.scatterplot(x="Age", y="Fare", hue="Sex", data=train);

plt.title('Fare and Age of Titanic Passengers by Sex', fontsize=18);

plt.xlabel('Age', fontsize=16);

plt.ylabel('Fare', fontsize=16);