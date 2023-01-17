import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

pd.set_option("display.precision", 2)
data = pd.read_csv('../input/titanic_train.csv',

                  index_col='PassengerId')
data.head(5)
data.describe()
data[(data['Embarked'] == 'C') & (data.Fare > 200)].head()
data[(data['Embarked'] == 'C') & 

     (data['Fare'] > 200)].sort_values(by='Fare',

                               ascending=False).head()
def age_category(age):

    '''

    < 30 -> 1

    >= 30, <60 -> 2

    >= 60 -> 3

    '''

    if age < 30:

        return 1

    elif age < 60:

        return 2

    else:

        return 3
age_categories = [age_category(age) for age in data.Age]

data['Age_category'] = age_categories
data['Age_category'] = data['Age'].apply(age_category)

data
data.groupby(by='Sex').count()
data.groupby(by=['Pclass','Sex']).count()
print(data['Fare'].median())

print(data['Fare'].std())

data.pivot_table(['Age'], ['Survived'], aggfunc='mean')
young_survived = data.loc[data['Age'] < 30, 'Survived']

old_survived = data.loc[data['Age'] > 60, 'Survived']



print("Young survived: {}%".format(round(100 * young_survived.mean(), 1)))

print("Old survived: {}%".format(round(100 * old_survived.mean(), 1)))

women_survived = data.loc[data['Sex'] == 'female', 'Survived']

men_survived = data.loc[data['Sex'] == 'male', 'Survived']



print("Women survived {}% of the time".format(round(100 * women_survived.mean(),1)))

print("Men survived {}% of the time".format(round(100 * men_survived.mean(),1)))
data.loc[1, 'Name'].split(',')[1].split()[1] #Take one name as example

names = data.loc[data['Sex'] == 'male', 'Name'].apply(lambda name: name.split(',')[1].split()[1])

names.value_counts().head()
data.groupby(['Sex', 'Pclass'])['Age'].mean()