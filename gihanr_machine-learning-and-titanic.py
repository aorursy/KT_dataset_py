import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv('../input/train.csv')
data.count()
data.describe()
# survived_female_ages_age_NaN = data[(data['Survived'] == 1) & (data['Sex'] == 'female') & (~survived_female_ages['Age'].notna())]
# len(survived_female_ages_age_NaN)

#All females
female_ages = data[(data['Sex'] == 'female')]

#Survived females
# survived_female_ages = data[(data['Survived'] == 1) & (data['Sex'] == 'female')]
import seaborn as sns
#Survived female ages and ages with NaN
female_ages_survived_age = female_ages[(female_ages['Survived'] == 1) & (female_ages['Age'].notna())]
print(len(female_ages_survived_age))
female_ages_survived_age_NaN = female_ages[(female_ages['Survived'] == 1) & (~female_ages['Age'].notna())]
print(len(female_ages_survived_age_NaN))
def drawBoxplot(datacolumen):
    sns.set_style("whitegrid")
    sns.boxplot(x=datacolumen)
    sns.swarmplot(x=datacolumen, color='black')
drawBoxplot(female_ages_survived_age['Age'])
female_ages_survived_age['Age'].mean()
#Not Survived female ages and ages with NaN
female_ages_not_survived_age = female_ages[(female_ages['Survived'] == 0) & (female_ages['Age'].notna())]
print(len(female_ages_not_survived_age))
female_ages_not_survived_age_NaN = female_ages[(female_ages['Survived'] == 0) & (~female_ages['Age'].notna())]
print(len(female_ages_not_survived_age_NaN))
drawBoxplot(female_ages_not_survived_age['Age'])
female_ages_not_survived_age['Age'].mean()
data_replaced_ages = data.copy()
# replace NaN with survived females means
data_replaced_ages.loc[(data_replaced_ages['Survived'] == 1) & (data_replaced_ages['Sex'] == 'female') & (data_replaced_ages['Age'].isnull()), 'Age'] = female_ages_survived_age['Age'].mean()
#replace NaN with not survived females
data_replaced_ages.loc[(data_replaced_ages['Survived'] == 0) & (data_replaced_ages['Sex'] == 'female') & (data_replaced_ages['Age'].isnull()), 'Age'] = female_ages_not_survived_age['Age'].mean()
#draw survived males with age
drawBoxplot(data_replaced_ages[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 1) & (data_replaced_ages['Age'].notna())]['Age'])
#draw not survived males with age
drawBoxplot(data_replaced_ages[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 0) & (data_replaced_ages['Age'].notna())]['Age'])
#replace NaN with survived males
survive_males = data_replaced_ages[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 1) & (data_replaced_ages['Age'].notna())]
data_replaced_ages.loc[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 1) & (data_replaced_ages['Age'].isnull()), 'Age'] = survive_males['Age'].median()
#replace NaN with Not survived males
not_survived_males = data_replaced_ages[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 0) & (data_replaced_ages['Age'].notna())]
data_replaced_ages.loc[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 0) & (data_replaced_ages['Age'].isnull()), 'Age'] = not_survived_males['Age'].median()
data_replaced_ages.head(10)
# Analysis of based on Pclass
data_replaced_ages_females = data_replaced_ages[(data_replaced_ages['Sex'] == 'female')]
# survived females with pclass
data_replaced_ages_females.groupby(['Survived','Pclass']).size().plot.bar()
# Analysis of based on Pclass
data_replaced_ages_males = data_replaced_ages[(data_replaced_ages['Sex'] == 'male')]
pd.crosstab(data_replaced_ages_males['Survived'],data_replaced_ages_males['Pclass']).plot.bar()
import matplotlib.pyplot as plt
# check ages of survived and not survived females
plt.subplots(figsize=(15,10))
sns.swarmplot(x='Sex', y='Age', hue='Survived', data=data_replaced_ages[['Survived','Sex', 'Age']], dodge=True).set_title('Survival based on Age')
# check ages of survived and not survived females
plt.subplots(figsize=(30,8))
sns.swarmplot(x='Sex', y='SibSp', hue='Survived', data=data_replaced_ages[['Survived','Sex', 'SibSp']], dodge=True).set_title('Survival vs Siblings and spouse')
plt.subplots(figsize=(30,8))
sns.swarmplot(x='Sex', y='Parch', hue='Survived', data=data_replaced_ages[['Survived','Sex', 'Parch']], dodge=True).set_title('Survival vs parents and child')
plt.subplots(figsize=(15,10))
sns.swarmplot(x='Sex', y='Fare', hue='Survived', data=data_replaced_ages[['Survived','Sex', 'Fare']], dodge=True).set_title('Survival based on Fare')
# Analysis on Cabins
data_cabins = data_replaced_ages[data_replaced_ages['Cabin'].notna()]
pd.crosstab(data_cabins['Sex'], data_cabins['Survived']).plot.bar()
data_cabins_null = data_replaced_ages[data_replaced_ages['Cabin'].isnull()]
pd.crosstab(data_cabins_null['Sex'], data_cabins_null['Survived']).plot.bar()
data_replaced_ages.loc[data_replaced_ages['Cabin'].notna(), 'Cabin_Status'] = 1
data_replaced_ages.loc[data_replaced_ages['Cabin'].isnull(), 'Cabin_Status'] = 0
data_replaced_ages.head()
#Analysis on survival based on Embarked
#data_replaced_ages.loc[data_replaced_ages['Embarked'].isnull()]
tmp = data_replaced_ages[(data_replaced_ages['Sex'] == 'male')]
pd.crosstab(tmp['Survived'],tmp['Embarked']).plot.bar().set_title('Male survival analysis with Embarked')

tmp1 = data_replaced_ages[(data_replaced_ages['Sex'] == 'female')]
pd.crosstab(tmp1['Survived'],tmp1['Embarked']).plot.bar().set_title('Female survival analysis with Embarked')
# data_replaced_ages.loc[data_replaced_ages['Embarked'].isnull()] = 'S'
data_replaced_ages.loc[data_replaced_ages['Embarked'].isnull(), 'Embarked'] = 'S'
from sklearn.datasets import load_iris
import sklearn.model_selection as model_selection
final_data_set = data_replaced_ages.drop(['Name', 'Ticket', 'Cabin'], 1)
test_dataset = pd.read_csv('../input/test.csv')
test_dataset.count()
test_dataset.loc[(test_dataset['Age'].isnull()) & (test_dataset['Sex'] == 'male'), 'Age'] = (not_survived_males['Age'].median() + survive_males['Age'].median())/2
test_dataset.loc[(test_dataset['Age'].isnull()) & (test_dataset['Sex'] == 'female'), 'Age'] = (female_ages_survived_age['Age'].mean() + female_ages_not_survived_age['Age'].mean())/2
test_dataset.loc[test_dataset['Cabin'].notna(), 'Cabin_Status'] = 1
test_dataset.loc[test_dataset['Cabin'].isnull(), 'Cabin_Status'] = 0
test_dataset.loc[test_dataset['Fare'].isnull(), 'Fare'] = test_dataset['Fare'].mean()
final_test_data_set = test_dataset.drop(['Name', 'Ticket', 'Cabin'], 1)
final_test_data_set.head()
final_data_set.head()
X_train = final_data_set.iloc[:,2:]
Y_train = final_data_set.iloc[:,1]
X_test = final_test_data_set.iloc[:,1:]
from sklearn import linear_model
# Change categorical values to numberical values using sklearn
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X_train['Sex_Label'] = le.fit_transform(X_train['Sex'])
X_train['Embarked_Label'] = le.fit_transform(X_train['Embarked'])
X_train.head()                
X_train = X_train.drop(['Sex', 'Embarked'], axis=1)
X_test['Sex_Label'] = le.fit_transform(X_test['Sex'])
X_test['Embarked_Label'] = le.fit_transform(X_test['Embarked'])
X_test = X_test.drop(['Sex', 'Embarked'], axis=1)
X_test.head()
X_train.head()
model = linear_model.LogisticRegression()
model.fit(X_train, Y_train)
nparray = model.predict(X_test)
results = pd.DataFrame(nparray)
results.count()
results.columns = ['Survived']
passengerId = final_test_data_set[['PassengerId']]
passengerId.count()
# pd.DataFrame({"PassengerId": passengerId, "Survived" : results }, ignore_index=True)
submit = pd.concat([passengerId, results], axis=1)
submit.to_csv('Submission.csv', index=False)
