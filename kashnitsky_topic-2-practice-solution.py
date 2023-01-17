import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
train_df = pd.read_csv("../input//titanic_train.csv", index_col='PassengerId') 
train_df.head(2)
train_df.describe(include='all')
train_df.info()
train_df = train_df.drop('Cabin', axis=1).dropna()
train_df.shape
sns.pairplot(train_df[['Survived', 'Age', 'Fare', 'SibSp', 'Parch']]);
sns.boxplot(x='Pclass', y='Fare', data=train_df);
sns.boxplot(x='Pclass', y='Fare', data=train_df[train_df['Fare'] < train_df['Fare'].quantile(.95)]);
pd.crosstab(train_df['Sex'], train_df['Survived'])
sns.countplot(x="Sex", hue="Survived", data=train_df);
sns.countplot(x="Pclass", hue="Survived", data=train_df);
sns.boxplot(x='Survived', y='Fare', data=train_df[train_df['Fare'] < 500]);
sns.boxplot(x='Survived', y='Age', data=train_df);
sns.boxplot(x='Survived', hue='Pclass', y='Age', data=train_df);
train_df['age_cat'] = train_df['Age'].apply(lambda age: 1 if age < 30 
                                            else 3 if age > 55 else 2);
pd.crosstab(train_df['age_cat'], train_df['Survived'])
sns.countplot(x='age_cat', hue='Survived', data=train_df);