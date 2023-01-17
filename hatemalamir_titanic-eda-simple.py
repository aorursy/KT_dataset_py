import os
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
INPUT_DIR = '/kaggle/input/titanic'
OUTPUT_DIR = 'output'
train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'),
                    index_col="PassengerId")

print(train.columns)
train.head(10)
train.isnull().sum()
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Cabin'].fillna('NA', inplace=True)
train['Embarked'].fillna('NA', inplace=True)
# Representing socio-economic status with numbers can be misleading for the model. It's better represented as a nominal variable
train['Pclass'] = train['Pclass'].map({1: "1st", 2: "2nd", 3: "3rd"})
train.groupby(['Pclass', 'Survived'])['Pclass'].count().unstack('Survived').plot.bar()
# Usually, names doesn't have obvious patterns to take into consideration in making predicions
print(train['Name'].nunique())
del train['Name']
train.groupby(['Sex', 'Survived'])['Sex'].count().unstack('Survived').plot.bar(stacked=True)
train.boxplot(column='Age', by='Survived')
sns.set(style="whitegrid")
sns.violinplot(x='Survived', y='Age', data=train)
fig = train.groupby('Survived')['SibSp'].plot.hist()
train.groupby('Survived')['Parch'].plot.hist()
train.groupby('Ticket')['Survived'].count().sort_values(ascending=False).head(20)
train.groupby('Ticket')['Survived'].sum().sort_values(ascending=False).head(20)
train.boxplot(column='Fare', by='Survived')
train.groupby('Cabin')['Survived'].count().sort_values(ascending=False).head(10)
train.groupby(['Embarked', 'Survived'])['Embarked'].count().unstack('Survived').plot.bar(stacked=True)
sns.boxplot(x="Pclass", y="Fare", data=train,hue="Survived", order=["1st", "2nd", "3rd"])
pd.pivot_table(train, values="Survived", index=["SibSp", "Parch"], aggfunc=np.sum)
emp_class_cnt = train.groupby(["Pclass", "Embarked"])["Survived"].count().unstack("Embarked")
sns.heatmap(emp_class_cnt, annot=True, cmap="YlGnBu")
emp_class_surv = train.groupby(["Pclass", "Embarked"])["Survived"].sum().unstack("Embarked")
sns.heatmap(emp_class_surv, annot=True, cmap="YlGnBu")
