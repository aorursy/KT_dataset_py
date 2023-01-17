import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
        
titanic = pd.read_csv("/kaggle/input/titanic/train.csv")
titanic.head()
#First of all we need to know the type and number of features for each Passanger
titanic.info()
titanic.groupby('Pclass', as_index=False)['Survived'].mean()
titanic.groupby('Sex', as_index=False)['Survived'].mean()
titanic.groupby('Embarked', as_index=False)['Survived'].mean()
#create a new column with the first letter of the column 'Cabin'
titanic['Cabin_first_letter'] = titanic['Cabin'].str[0]
#replace NAN values with 'other'
titanic['Cabin_first_letter'].fillna('other', inplace = True)
titanic.head()
#group the number of survived per 'Cabin' letters
gp_survived_cabin = titanic[['Cabin_first_letter', 'Survived']].groupby(['Cabin_first_letter'], as_index=False).count()
gp_survived_cabin
#Plot the distribution of survivers 
facet = sns.FacetGrid(data = titanic, hue = "Cabin_first_letter", legend_out=True, height= 5)
facet = facet.map(sns.kdeplot, "Age")
facet.add_legend();
#Thanks to : https://www.kaggle.com/kpacocha/top-5-titanic-machine-learning-from-disaster

titanic.Fare = titanic.Fare.fillna(0)
titanic.Embarked = titanic.Embarked.fillna('S')

titanic.Cabin = titanic.Cabin.fillna('Unknown_Cabin')
titanic['Cabin'] = titanic['Cabin'].str[0]

#I'm not going to change the unknown cabins for any letter, I think this will fit better.

titanic.info()
#Thanks to : https://www.kaggle.com/kpacocha/top-5-titanic-machine-learning-from-disaster

titanic['Title'] = titanic.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
