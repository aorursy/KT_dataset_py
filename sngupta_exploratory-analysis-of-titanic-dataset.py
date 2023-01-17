import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid')

import warnings
warnings.simplefilter('ignore')
train= pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')
print('Shape of train data: {}' .format(train.shape))
print('Shape of test data: {}' .format(test.shape))
train_= train.drop(['PassengerId'], axis= 1)
test_= test.drop(['PassengerId'], axis= 1)
print('Description of train data: ')
print(train_.describe())
print()
print('Description of test data: ')
print(test_.describe())
train_.head()
test_.head()
null_value= pd.DataFrame({
    'Total_null_train': train_.isnull().sum(),
    'Percent_null_train': train_.isnull().sum()/train_.shape[0],
    'Total_null_test': test_.isnull().sum(),
    'Percent_null_test': test_.isnull().sum()/test_.shape[0]
})
null_value
for ind in null_value.index:
    print()
    print('The detail of feature: {}' .format(ind))
    print(train[ind].describe())
plt.hist(train['Survived'], color= 'red')
plt.show()
print('Analysis of the label feature: Survived on the basis of Sex')
sns.countplot('Survived', data= train_, hue= 'Sex')
plt.title('Survived on the basis of Sex')
plt.show()
sns.countplot('Survived', data= train_, hue= 'Pclass')
plt.title('Survived on the basis of Paseenger Class')
sns.countplot('Survived', data= train_, hue= 'Embarked')
plt.title('Survived on the basis of Emabarked')
sns.countplot('Survived', data= train_, hue= 'SibSp')
plt.title('Survived on the basis of Sibling and spouse')
plt.legend(loc= 'best')
plt.scatter(np.log(train['Fare']), train['Survived'])
sns.lmplot('Fare', 'Survived', data= train_)
plt.figure(figsize= (8,6))
sns.heatmap(train_.corr(), annot= True, cmap= 'viridis')
null_value.index
train['Age'].groupby(train['Pclass']).describe()
sns.boxplot('Pclass', 'Age', data= train_, hue= 'Sex', )
plt.title('Boxplot Pclass vs Age')
age_bin= pd.cut(train['Age'], bins=[1,10,20,30,40,50,60,70,80,90,100])
age_bin.value_counts()
age_bin.value_counts().plot(kind= 'pie')
sns.boxplot('Pclass', 'Age', data= train)
sns.boxplot('Embarked', 'Age', data= train)
print('Average age of Pclass: {} and Sex: {}' .format(1, 'male'))
print(train[(train['Pclass']==1) & (train['Sex']=='male')]['Age'].mean())

print('Average age of Pclass: {} and Sex: {}' .format(1, 'female'))
print(train[(train['Pclass']==1) & (train['Sex']=='female')]['Age'].mean())

print('Average age of Pclass: {} and Sex: {}' .format(2, 'male'))
print(train[(train['Pclass']==2) & (train['Sex']=='male')]['Age'].mean())

print('Average age of Pclass: {} and Sex: {}' .format(2, 'female'))
print(train[(train['Pclass']==2) & (train['Sex']=='female')]['Age'].mean())

print('Average age of Pclass: {} and Sex: {}' .format(3, 'male'))
print(train[(train['Pclass']==3) & (train['Sex']=='male')]['Age'].mean())

print('Average age of Pclass: {} and Sex: {}' .format(3, 'female'))
print(train[(train['Pclass']==3) & (train['Sex']=='female')]['Age'].mean())
train_['Age']= train_['Age'].fillna(train_['Age'].mode()[0])
test_['Age']= test_['Age'].fillna(test_['Age'].mode()[0])
train_['Embarked']= train_['Embarked'].fillna(train_['Embarked'].mode()[0])
train_['Cabin']= train_['Cabin'].fillna('None')
test_['Cabin']= test_['Cabin'].fillna('None')
test_['Fare']= test_['Fare'].fillna(test_['Fare'].mean())
train_['Cabin']= [x[:1] for x in train_['Cabin']]
test_['Cabin']= [x[:1] for x in test_['Cabin']]
Name= list(train_['Name'])
Name_= list(test_['Name'])
salutation= []
for name in Name:
    last= name.split(',')[1]
    sal= last.split('.')[0]
    salutation.append(sal)

sal_test= []
for name_ in Name_:
    last_= name_.split(',')[1]
    sal_= last_.split('.')[0]
    sal_test.append(sal_)
type(salutation)
train_['Name']= [x for x in salutation]
test_['Name']= [x for x in sal_test]
train_['Name'].value_counts()
train_['Name']= [x.replace('Miss', 'Mrs') for x in train_['Name']]
train_['Name']= [x.replace('Master', 'Mr') for x in train_['Name']]
train_['Name']= [x.replace('Mlle', 'Mrs') for x in train_['Name']]
train_['Name']= [x.replace('Mme', 'Mrs') for x in train_['Name']]
train_['Name']= [x.replace('Dr', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Rev', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Col', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Major', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Don', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Capt', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Sir', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Jonkheer', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('the Countess', 'Other') for x in train_['Name']]
train_['Name']= [x.replace('Lady', 'Mrs') for x in train_['Name']]
train_['Name']= [x.replace('Ms', 'Other') for x in train_['Name']]
test_['Name'].value_counts()
test_['Name']= [x.replace('Miss', 'Mrs') for x in test_['Name']]
test_['Name']= [x.replace('Master', 'Mr') for x in test_['Name']]
test_['Name']= [x.replace('Rev', 'Other') for x in test_['Name']]
test_['Name']= [x.replace('Col', 'Other') for x in test_['Name']]
test_['Name']= [x.replace('Dona', 'Other') for x in test_['Name']]
test_['Name']= [x.replace('Ms', 'Other') for x in test_['Name']]
test_['Name']= [x.replace('Dr', 'Other') for x in test_['Name']]
train_['Name'].value_counts()
test_['Name'].value_counts()
train_.isnull().sum()
test_.isnull().sum()