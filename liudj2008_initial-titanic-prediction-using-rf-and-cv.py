import numpy as np

import sklearn

import pandas as pd

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
# Import sklearn modules

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df.shape
df.info()
df.head()
# Use sex as one metric, because being female has significant higher survival rate

sns.catplot(x='Sex', y='Survived', data=df, kind='bar')
df['Sex'] = df['Sex'].apply(lambda x: 0 if x=='male' else 1)
df.loc[df['Age'].isnull(), 'Age'] = df['Age'].median()
# Define function that divide age into three group: <18 juniors; 18-56 Adults; >56 seniors

age_to_category = lambda x: 1 if x<18 else 2 if x<56 else 3
df['Age'] = df['Age'].apply(age_to_category)
# Juniors have higher survival rate than adult and seniors, and this could be included into the metrics

sns.catplot(x='Age', y='Survived', data=df, kind='bar')
# Fare price has been heavily right skewed, with mean>median

plt.figure(figsize=(20,10))

sns.distplot(df['Fare'], kde=False)

plt.show()
# Define three category of fares

fare_to_category = lambda x: 1 if x<20 else 2 if x<100 else 3
df['Fare'] = df['Fare'].apply(fare_to_category)
# Higher fare class has much higher survival rate than lower fare class

sns.catplot(x='Fare', y='Survived', data=df, kind='bar')
sns.catplot(x='Embarked', y='Survived', data=df, kind='bar')
df['Embarked'].fillna(np.random.choice(df.dropna(subset=['Embarked'])['Embarked']), inplace=True)
df['Embarked'] = df['Embarked'].apply(lambda x: 0 if x=='S' else 1 if x=='C' else 2)
df[['Pclass', 'Age', 'SibSp','Parch', 'Fare']].corr()
# parent and children (Parch) and SibSp(siblings and spouse) are positive correlated, higher parch means higher sibsp

# Fare and pclass are negatively correlated. Upper class (1) means higher fare price

sns.heatmap(df[['Pclass', 'Age', 'SibSp','Parch', 'Fare']].corr())
sns.catplot(x='Pclass', y='Survived', data=df, kind='bar')
df['FamilyMember'] = df['Parch'] + df['SibSp']
# 3 Family members reached the highest survival rate

sns.catplot(x='FamilyMember', y='Survived', data=df, kind='bar')
df = df[['Pclass', 'Sex', 'Age', 'FamilyMember', 'Fare', 'Embarked', 'Survived']]

df.head()
# Seperate train and test data using stratified method

x_train, x_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), 

                                                    df['Survived'], 

                                                    test_size=0.3, 

                                                    random_state=1,

                                                    stratify=df['Survived'])
# Prediction accuracy using sex as the only feature

(x_test['Sex']==y_test).mean()
# Generate benchmark submission using sex as the only feature

df_test['Survived'] = df_test['Sex']

df_test[['PassengerId', 'Survived']].to_csv('submission_sex.csv', encoding='utf8', index=False)
# Use random forest as classifier, and optimize n_estimators

rf = RandomForestClassifier(random_state=1)

parameters = {'n_estimators':[10, 20, 50, 100]}
# Grid search for the best number of estimators

clf = GridSearchCV(rf, parameters, cv=5)
clf.fit(x_train, y_train)
clf.best_score_
clf.best_params_
(clf.predict(x_test)==y_test).mean()
confusion_matrix(y_test, clf.predict(x_test))
roc_auc_score(y_test, clf.predict(x_test))
# Processing test data for submission

df_test['FamilyMember'] = df_test['SibSp']+df_test['Parch']



df_test.loc[df_test['Age'].isnull(), 'Age'] = df_test['Age'].median()



df_test['Age'] = df_test['Age'].apply(age_to_category)



df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)



df_test['Fare'] = df_test['Fare'].apply(fare_to_category)



df_test['Embarked'] = df_test['Embarked'].apply(lambda x: 0 if x=='S' else 1 if x=='C' else 2)



df_test['Sex'] = df_test['Sex'].apply(lambda x: 0 if x=='male' else 1)
df_test = df_test[['PassengerId', 'Pclass', 'Sex', 'Age', 'FamilyMember', 'Fare', 'Embarked']]
df_test['Survived'] = clf.predict(df_test.drop('PassengerId', axis=1))
# Generate submission file

df_test[['PassengerId', 'Survived']].to_csv('submission_cv.csv', encoding='utf8', index=False)