# linear algebra

import numpy as np 



# # data processing

import pandas as pd



# data visualization

import seaborn as sns

from matplotlib import pyplot as plt

from matplotlib import style

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
# looking at first 5 rows 

train_df.head(5)
# checking the data for null values

sns.heatmap(train_df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
# setting a consitent style from seaborn styles: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'

sns.set_style('whitegrid') 
sns.countplot(x='Survived', data=train_df, hue='Sex')
sns.countplot(x='Survived', data=train_df, hue='Pclass')
sns.distplot(train_df['Age'].dropna(), kde=False, bins=30)
sns.countplot(x='SibSp', data=train_df)
train_df['Fare'].hist(bins=40)
import cufflinks as cf

cf.go_offline()
train_df['Fare'].iplot(kind='hist', bins=30, color='green')
sns.boxplot(x='Pclass', y='Age', data=train_df)
mean_age_pclass1 = round(train_df['Age'][train_df['Pclass']==1].dropna().mean())

mean_age_pclass2 = round(train_df['Age'][train_df['Pclass']==2].dropna().mean())

mean_age_pclass3 = round(train_df['Age'][train_df['Pclass']==3].dropna().mean())



print('Mean age in 1st Pclass: {}'.format(mean_age_pclass1))

print('Mean age in 2nd Pclass: {}'.format(mean_age_pclass2))

print('Mean age in 3rd Pclass: {}'.format(mean_age_pclass3))
# replace age with mean value of according Pclass

def replace_nan_by_mean_age(cols):

    age = cols[0]

    pclass = cols[1]

    

    if pd.isnull(age):

        if pclass == 1:

            return mean_age_pclass1

        elif pclass == 2:

            return mean_age_pclass2

        else:

            return mean_age_pclass3

    else:

        return age
train_df['Age'] = train_df[['Age', 'Pclass']].apply(replace_nan_by_mean_age, axis=1)
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train_df.drop('Cabin', axis=1, inplace=True)
train_df.dropna(inplace=True)
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train_df.info()
sex = pd.get_dummies(train_df['Sex'], drop_first=True)

embark = pd.get_dummies(train_df['Embarked'], drop_first=True)

train_df = pd.concat([train_df, sex, embark], axis=1)
train_df.head()
train_df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train_df.head()
# Algorithms

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(train_df.drop('Survived', axis=1), train_df['Survived'], test_size=0.3, random_state=101)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
predictions = log_model.predict(X_test)
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test, predictions))