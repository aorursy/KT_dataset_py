import numpy as np

import pandas as pd 

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
print('Test data shape: ', train.shape)

train.head()
train.describe()
train.info()
train.isnull().sum()
test.isnull().sum()
imputer = SimpleImputer(np.nan, "mean")



train['Age'] = imputer.fit_transform(np.array(train['Age']).reshape(891, 1)) 

train.Embarked.fillna(method='ffill', inplace=True) 

train.drop(['PassengerId', 'Name','Ticket'], axis=1, inplace=True)

train.head()
test['Age'] = imputer.fit_transform(np.array(test['Age']).reshape(418, 1))

test.Embarked.fillna(method='ffill', inplace=True)

test.Fare.fillna(method='ffill', inplace=True)

test.drop(['Name', 'Ticket'], axis=1, inplace=True)

test.head()
train['Survived'].value_counts()
plt.figure(figsize=[10,5])

sns.countplot(x = 'Sex', hue = 'Survived', data = train)

plt.xticks(rotation = 20);
sns.barplot(x='Sex', y='Survived', data=train, palette=('RdPu'));
print('% of survived females:', train['Survived'][train['Sex'] == 'female'].value_counts(normalize = True)[1]*100)

print('% of survived males:', train['Survived'][train['Sex'] == 'male'].value_counts(normalize = True)[1]*100)
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
fig, ax = plt.subplots(figsize = (10,6))

ax = sns.countplot(x = 'Survived', hue = 'Pclass', data = train, palette = 'YlOrRd')

ax.set_xlabel('Survived')

ax.set_title('Survival Rate for Passenger Classes', fontsize = 14, fontweight='bold');
ax = sns.catplot(x="Pclass", hue="Sex", col="Survived",

                data=train, kind="count",

                height=4, aspect=.7, palette = 'OrRd');
sns.countplot(x = "Pclass", hue = "Survived", data = train, palette = 'RdPu');
sns.barplot(x="Pclass", y="Survived", data= train, palette = 'BuGn');
perc = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

perc*100
sns.factorplot(x='Pclass', y='Survived', hue = 'Sex', data = train, palette = 'PRGn');
sns.boxplot(x='Sex', y='Age', hue = 'Survived',data=train);
grid = sns.FacetGrid(train, col='Survived')

grid.map(plt.hist, 'Age', bins=25, color = 'y').add_legend();

sns.set(style="ticks", color_codes=True);
plt.figure(figsize=(10,5))

sns.distplot(train['Age'], bins=24, color='g');
avg_age_train = train ["Age"].mean()

std_age_train = train ["Age"].std()



avg_age_test = test["Age"].mean()

std_age_test = test ["Age"].std()
bins = [0, 1, 12, 18, 21,  60, np.inf]

labels = ['Infant', 'Child', 'Teenager',' Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

sns.barplot(x="AgeGroup", y="Survived", data= train)

plt.show;
train.head()
train[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='Survived', ascending=False)
AgeGroup_train  = pd.get_dummies(train['AgeGroup'])

AgeGroup_train.columns = ['Infant', 'Child', 'Teenager',' Young Adult', 'Adult', 'Senior']

AgeGroup_test  = pd.get_dummies(test['AgeGroup'])

AgeGroup_test.columns = ['Infant', 'Child', 'Teenager',' Young Adult', 'Adult', 'Senior']



train = train.join(AgeGroup_train)

test = test.join(AgeGroup_test)
age_mapping = {'Infant': 1, 'Child': 2, 'Teenager': 3, 'Young Adult': 4, 'Adult': 5, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)



train.head()

train.drop(['AgeGroup', 'Age'],axis=1,inplace=True)

test.drop(['AgeGroup', 'Age'],axis=1,inplace=True)

train.head()
train["Cabin_new"] = (train["Cabin"].notnull().astype('int'))

test["Cabin_new"] = (test["Cabin"].notnull().astype('int'))

train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)

print("% of Cabin = 1 survived:", train["Survived"][train["Cabin_new"] == 1].value_counts(normalize = True)[1]*100)

print("% of Cabin = 0 survived:", train["Survived"][train["Cabin_new"] == 0].value_counts(normalize = True)[1]*100)



sns.barplot(x="Cabin_new", y="Survived", data=train).set_title('Cabin vs No Cabin')

plt.show()
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train['FamilySize'] = train['Parch'] + train['SibSp']

test['FamilySize'] = test['Parch'] + test['SibSp']

#train.drop(['Parch', 'SibSp'], axis=1,inplace=True)

#test.drop(['Parch', 'SibSp'], axis=1,inplace=True)

sns.barplot(x="FamilySize", y="Survived", data=train)

plt.show;
sns.pointplot(x='FamilySize', y = 'Survived', data = train);
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q']);
sns.countplot(x='Embarked', hue='Survived', data=train);
embark_dummies_train  = pd.get_dummies(train['Embarked'])

embark_dummies_test  = pd.get_dummies(test['Embarked'])



train = train.join(embark_dummies_train)

test = test.join(embark_dummies_test)

train.head()
train = train.drop(['Embarked'], axis = 1)

test = test.drop(['Embarked'], axis = 1)

train.head()
sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



train.head()
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True)

sns.kdeplot(train['Survived'], train['Sex'],cbar = cmap,shade=True);
g = sns.jointplot(x='Survived', y = 'Sex', data=train, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("Survived", "Gender")
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(train['Survived'], train['Sex'], ax=ax)

sns.rugplot(train['Survived'], color="g", ax=ax)

sns.rugplot(train['Sex'], vertical=True, ax=ax);
# convert from float to int

train['Fare'] = train['Fare'].astype(int)

test['Fare']= test['Fare'].astype(int)



# get fare for survived & didn't survive passengers 

fare_not_survived = train["Fare"][train["Survived"] == 0]

fare_survived = train["Fare"][train["Survived"] == 1]



# get average and std for fare of survived/not survived passengers

avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])



# plot

train['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50), color = 'purple')



avgerage_fare.index.names = std_fare.index.names = ["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False, color = 'r');
mask = np.zeros_like(train.corr(), dtype=np.bool)

## in order to reverse the bar replace "RdBu" with "RdBu_r"

plt.subplots(figsize = (15,12))

sns.heatmap(train.corr(), annot=True,mask = False,cmap = 'OrRd', linewidths=.7, linecolor='black',fmt='.2g',center = 0,square=True)



plt.title("Correlations", y = 1.03,fontsize = 20, fontweight = 'bold', pad = 40);