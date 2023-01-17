# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

# Figures inline and set visualization style
%matplotlib inline
sns.set()
# Import test and train datasets
df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')

# View first lines of training data
df_train.head(15)
df_test.head()
# View first lines of test data
df_test.head(10)
df_train.info()
df_train.describe()

sns.countplot(x='Survived', data=df_train)
df_test['Survived'] = 0
df_test[['PassengerId', 'Survived']].to_csv('no_survivors.csv', index=False)
sns.countplot(x='Sex', data=df_train)
sns.catplot(x='Survived', col='Sex', kind='count', data=df_train)
df_train.groupby(['Sex']).Survived.sum()
print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())
df_test['Survived'] = df_test.Sex == 'female'
df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test.head()
df_test[['PassengerId', 'Survived']].to_csv('women_survive.csv', index=False)
sns.catplot(x='Survived', col='Pclass', kind='count', data=df_train);
sns.catplot(x='Embarked', col='Pclass', kind='count', data=df_train);
sns.catplot(x='Fare', col='Survived', kind='count', data=df_train);
