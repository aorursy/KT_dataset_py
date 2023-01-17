# Importing necessary packages

import pandas as pd

from pandas import Series,DataFrame
titanic_df = pd.read_csv('../input/train.csv')
titanic_df.head()
titanic_df.info()
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df.describe()
pd.crosstab(titanic_df['Survived'],titanic_df['Sex'],margins = True)
pd.crosstab(titanic_df['Pclass'],titanic_df['Sex'],margins=True)
pd.crosstab([titanic_df.Pclass,titanic_df.Sex],titanic_df.Survived,margins=True)
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Males vs Females based on count on the Titanic

a1 = sns.factorplot(x = 'Sex',data=titanic_df, kind = 'count')

sns.plt.suptitle("COUNT OF MALES VS FEMALES ONBOARD", fontsize = 16)

a1.fig.subplots_adjust(top=.8)
# Classification of class of males and females. It appears that a majority of males and females 

# belonged to the third class 

print(titanic_df.columns)

sns.factorplot(x = 'Sex',data=titanic_df, kind ='count', hue = 'Pclass')
# Representation of survival amongst males and females. Looks like a major portion of females survived than males.

sns.factorplot(x = 'Sex', data = titanic_df, kind = 'count', hue = 'Survived')
sns.factorplot(data=titanic_df, x='Pclass',kind='count',hue='Survived')
### Breaking down the types of people into Male, Male child, Female and Female child
def male_female_child(passenger):

    age,Sex = passenger

    if age < 16 and Sex == 'female':

        return 'Female Child'

    elif age < 16 and Sex == 'male':

        return 'Male Child'

    else :

        return Sex
titanic_df['Person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
print(titanic_df.columns)
sns.factorplot('Person', data = titanic_df, kind = 'count')
sns.factorplot(x='Person',data = titanic_df , kind = 'count', hue='Survived')
sns.factorplot(x='Pclass',data = titanic_df , kind = 'count', hue='Survived')
sns.factorplot(x='Person', y = 'Survived',data = titanic_df , kind='bar', col='Pclass')
titanic_df.columns
#sns.factorplot(x='Embarked', y='Survived',hue='Sex', data=titanic_df, kind='bar', palette='muted')

sns.countplot(x='Embarked',hue='Survived', data=titanic_df)
pd.crosstab(titanic_df.Embarked,titanic_df.Survived,margins=True)
titanic_df['Name'].head(10)
import re



#### Define function to extract titles from Names using Regex

def extract_titles(Name):

    title = re.search('\s([A-Za-z]+)',Name)

    if title:

        return title.group(1)

    return ""

print(extract_titles("Braund, Mr. Owen Harris"))
# Get list of Titles

titles = titanic_df['Name'].apply(extract_titles)
# List the values that we have obtained and their count

pd.value_counts(titles)
# Create a dictionary with titles that we could consider

relevant_titles = {"Mr": 1, "Miss": 1, "Mrs": 1, "Master": 1, "Dr": 1, "Rev": 1, "Major": 1, "Col": 1, 

                 "Mlle": 1, "Mme": 1, "Don": 1, "Lady": 1, "Countess": 1, "Jonkheer": 1, "Sir": 1,

                 "Capt": 1, "Ms": 1}
# Convert each relevant title into a number

for k,v in relevant_titles.items():

    titles[titles == k] = v
# Add titles back to the DataFrame

titanic_df['Titles_that_matter'] = titles
titanic_df.head(5)
def is_title_or_not(Titles_with_N):

    if Titles_with_N == 1:

        return('Yes')

    return('No')
titanic_df['Title'] = titanic_df['Titles_that_matter'].apply(is_title_or_not)
titanic_df.head(3)
pd.value_counts(titanic_df['Title'])
sns.factorplot('Sex',data = titanic_df, kind = 'count', hue = 'Title')

# A large portion of males were with titles
sns.factorplot('Title',data = titanic_df, kind = 'count', hue = 'Survived')

# A large portion of people with titles did not survive which was expected as only 25 people are without titles
sns.factorplot(x= 'Sex', y = 'Survived', data = titanic_df, hue = 'Title', kind = 'bar')

# Females/Males without titles survived more than females with titles.

sns.factorplot(x= 'Title', y = 'Survived', data = titanic_df, hue = 'Pclass', kind = 'bar')

# Those with titles in the first class survived more than the second and third class