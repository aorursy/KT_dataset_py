#You're usual round of imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Importing the data
data = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#Looking at the data to see what we have.
data.head()
#I'm dropping These columns since they don't seem like they will be 
#useful in predicting survival.

data = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1)
#looking at the Test data.
test.head()
#Again, dropping columns that don't seem useful.
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#I wanted to take a look at the shorter list of columns I
#plan on working with.
data.head()
#I wanted to see how many males and females there were, and
#get an idea of what the survival rates for the respective groups were
%matplotlib inline

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

sns.barplot(x='Sex', y='Survived', data=data, ax=ax1)
ax1.set_ylabel('Survival based on gender')
ax1.set_title('Survival percentaged based on gender')
ax1.set_xlabel('')
ax1.set_ylabel('')

sns.countplot(data['Sex'], ax=ax2)
ax2.set_title('Count of each gender')
ax2.set_ylabel('')
ax2.set_xlabel('')
#There are very few males who survived and very few females who didn't 
#survive. I want to look at survival based on sex for various age ranges. 

g = sns.FacetGrid(data, row='Sex', col='Survived', margin_titles=True)
bins = np.linspace(data['Age'].min(), data['Age'].max(), 20)
g.map(plt.hist, 'Age', bins=bins, lw=0)

#I want to take a closer look into men who did survive and women 
#who didn't. I want to look at the other potential factors to see
#if something stands out.
#Let's look at how class factors into survival for each gender.

g = sns.factorplot(data=data, x='Pclass', y='Survived', hue='Sex', kind='bar')
g.despine(left=True)
g.set_ylabels('Chance for Survival')

#Looks like women in 1st or 2nd class are basically guaranteed to survive, 
#and women in 3rd class are at chance. Men in 1st class are the most likely
#group to survive, and 2nd/3rd class men are very unlikely to survive.
#I want to take a look at how SibSp and gender affect survival.

#makes copies of the data based on sex
males = data.loc[data['Sex'] =='male']
females = data.loc[data['Sex'] =='female']

#plot for survival of males based on Class and the number of siblings
#they have
g1 = sns.FacetGrid(males, row='Pclass', col='Survived', margin_titles=True)
bins = np.linspace(males['SibSp'].min(), males['SibSp'].max(), 20)
g1.map(plt.hist, 'SibSp', bins=bins, lw=0)
plt.subplots_adjust(top=.93)
g1.fig.suptitle('Survival data for males based on class & SibSp')

#plot for survival of females based on Class and the number of siblings 
#they have
g2 = sns.FacetGrid(females, row='Pclass', col='Survived', margin_titles=True)
bins = np.linspace(females['SibSp'].min(), females['SibSp'].max(), 20)
g2.map(plt.hist, 'SibSp', bins=bins, lw=0)
plt.subplots_adjust(top=.93)
g2.fig.suptitle('Survival data for females based on class & SibSp')

#The number of siblings/spouse doesn't appear to have any strong
#relationships for males. This does seem to reinforce the idea that being
#in a lower class for males has a higher likely hood for death. 
#For females if you have more than 1 sibling/spouse your odds drop
#significantly if you're in 1st or 2nd class. 3rd class seems to be a wash.
#Since class seems to be a strong indicator for survival, let's see if 
#age within a given class has any interesting insights.

#plot for survival of males based on Class and age
g1 = sns.FacetGrid(males, row='Pclass', col='Survived', margin_titles=True)
bins = np.linspace(males['Age'].min(), males['Age'].max(), 20)
g1.map(plt.hist, 'Age', bins=bins, lw=0)
plt.subplots_adjust(top=.93)
g1.fig.suptitle('Survival data for males based on class and age')

#plot for survival females based on Class and age
g2 = sns.FacetGrid(females, row='Pclass', col='Survived', margin_titles=True)
bins = np.linspace(females['Age'].min(), females['Age'].max(), 20)
g2.map(plt.hist, 'Age', bins=bins, lw=0)
plt.subplots_adjust(top=.93)
g2.fig.suptitle('Survival data for females based on class and age')

#for males age doesn't appear to matter for 1st class, but for
#2nd and 3rd class the younger you are, the less likely you are to survive.

#for females if you're in 1st or 2nd class you're odds are good regardless
#of age. Females in 3rd class seem to have a 50/50 chance.
#Let's take a look at Parent/Children to see if anything stands out.

#makes copies of the data based on sex
males = data.loc[data['Sex'] =='male']
females = data.loc[data['Sex'] =='female']

#plot for males based on Class and the number of Parents/Children they have
g1 = sns.FacetGrid(males, row='Pclass', col='Survived', margin_titles=True)
bins = np.linspace(males['Parch'].min(), males['Parch'].max(), 20)
g1.map(plt.hist, 'Parch', bins=bins, lw=0)
plt.subplots_adjust(top=.93)
g1.fig.suptitle('Survival data for males based on class & Parch')

#plot for females based on Class and the number of Parents/Children they have
g2 = sns.FacetGrid(females, row='Pclass', col='Survived', margin_titles=True)
bins = np.linspace(females['Parch'].min(), females['Parch'].max(), 20)
g2.map(plt.hist, 'Parch', bins=bins, lw=0)
plt.subplots_adjust(top=.93)
g2.fig.suptitle('Survival data for females based on class & Parch')

#For males it appears there's no relationship for 1st class, 2nd class,
#has 
#Since class seems to be a strong indicator for survival, let's see if 
#age within a given class has any interesting insights.

#plot for males based on Class and the number of siblings they have
g1 = sns.FacetGrid(males, row='Parch', col='Survived', margin_titles=True)
bins = np.linspace(males['Age'].min(), males['Age'].max(), 20)
g1.map(plt.hist, 'Age', bins=bins, lw=0)
plt.subplots_adjust(top=.93)
g1.fig.suptitle('Survival data for males based on Parch and age')

#plot for females based on Class and the number of siblings they have
g2 = sns.FacetGrid(females, row='Parch', col='Survived', margin_titles=True)
bins = np.linspace(females['Age'].min(), females['Age'].max(), 20)
g2.map(plt.hist, 'Age', bins=bins, lw=0)
plt.subplots_adjust(top=.93)
g2.fig.suptitle('Survival data for females based on Parch and age')

#the answer seems to be the fewer you have the better off you are 
#this function adds a 1 if a passenger has family, and a 0 if not
def hasFamily(r):
    if r['SibSp'] >0 or r['Parch']>0:
        family = 1
    else:
        family = 0
    return family

data['HasFamily'] = data.apply(hasFamily, axis=1)
data = data.drop(['SibSp', 'Parch'], axis=1)

test['HasFamily'] = test.apply(hasFamily, axis=1)
test = test.drop(['SibSp', 'Parch'], axis=1)

#Let's see how survival changes if someone has family onboard
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

sns.barplot(x='HasFamily', y='Survived', data=data, ax=ax1)
ax1.set_ylabel('Survival based on has family')
ax1.set_title('Survival percentaged based on Family')
ax1.set_xlabel('')

sns.countplot(data['HasFamily'], ax=ax2)
ax2.set_title('Count of people with and without family')
ax2.set_xlabel('')
ax2.set_xticklabels(['No Family', 'Has Family'])
data = data.drop(['Embarked'], axis=1)
test = test.drop(['Embarked'], axis=1)
def genderNumber(r):
    if r['Sex'] == 'male':
        gender = 0
    else:
        gender = 1
    return gender

data['GenderNumber'] = data.apply(genderNumber, axis=1)
test['GenderNumber'] = test.apply(genderNumber, axis=1)
train_output = data['Survived']
train_features = data.drop(['Survived', 'Sex'], axis=1)


test_features = test.copy()
test_features = test_features.drop('Sex', axis=1)
train_features['Age'].fillna(test['Age'].median(), inplace=True)
test_features['Age'].fillna(test['Age'].median(), inplace=True)

train_features['Fare'].fillna(test['Fare'].median(), inplace=True)
test_features['Fare'].fillna(test['Fare'].median(), inplace=True)
train_features.head()
import sklearn 

from sklearn.linear_model import LogisticRegression
from sklearn import tree
r = LogisticRegression()

r.fit(train_features, train_output)
test_output = r.predict(test_features)

r.score(train_features, train_output)
tree = tree.DecisionTreeClassifier()

tree.fit(train_features, train_output)

test_output = tree.predict(test_features)

tree.score(train_features, train_output)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_output
    })
submission.to_csv('titanic.csv', index=False)