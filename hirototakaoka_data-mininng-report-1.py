import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd
train=pd.read_csv('../input/titanic/train.csv')

train.tail()
test=pd.read_csv('../input/titanic/test.csv')

test.isnull().sum()
train['Survived'].describe()
sns.countplot(train['Survived'],data=train);
sns.countplot(x='Pclass',hue='Survived',data=train);
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train);
sns.countplot(x='Sex',hue='Survived',data=train);
sns.countplot(x='Embarked',hue='Survived',data=train);
sns.countplot(x='Embarked',hue='Pclass',data=train);
sns.countplot(x='Embarked',hue='Sex',data=train);
sns.heatmap(train.isnull());
train['Age'].isnull().sum()
train['middle']=train['Name'].map(lambda x:x.split(',')[1].split('.')[0])

sns.countplot(x='middle',data=train);
test['middle']=test['Name'].map(lambda x:x.split(',')[1].split('.')[0])

test['middle'].value_counts()
train['middle'].value_counts()
train_Mr=train[train['middle'].str.contains('Mr')]

train_Mrs=train[train['middle'].str.contains('Mrs')]

train_Miss=train[train['middle'].str.contains('Miss')]

train_Master=train[train['middle'].str.contains('Master')]



print('Mr {}'.format(train_Mr['Age'].dropna().mean()))

print('Mrs {}'.format(train_Mrs['Age'].dropna().mean()))

print('Miss {}'.format(train_Miss['Age'].dropna().mean()))

print('Master {}'.format(train_Master['Age'].dropna().mean()))

print('All {}'.format(train['Age'].dropna().mean()))
for i in range(len(train)):

        

    if pd.isnull(train['Age'][i]):

        name=train['middle'][i]

        if name=='Mr':

            train['Age'][i]=33

        elif name=='Mrs':

            train['Age'][i]=36

        elif name=='Miss':

            train['Age'][i]=22

        elif name=='Master':

            train['Age'][i]=5

        else:

            train['Age'][i]=30



train['Age'].isnull().sum()
for i in range(len(test)):

    if pd.isnull(test['Age'][i]):

        name=test['middle'][i]

        if name=='Mr':

            test['Age'][i]=33

        elif name=='Mrs':

            test['Age'][i]=36

        elif name=='Miss':

            test['Age'][i]=22

        elif name=='Master':

            test['Age'][i]=5

        else:

            test['Age'][i]=30



test['Age'].isnull().sum()
train['Family']=train['SibSp']+train['Parch']+1
test['Family'] = test['SibSp'] + test['Parch'] + 1
sns.countplot(x='Family',hue='Survived',data=train);
train['Family']=pd.cut(train.Family, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])

test['Family']= pd.cut(test.Family, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])

train.head()
y=train['Survived']

features=['Pclass','Sex','Age','Fare','Embarked','Family']

X=train[features]

X.tail()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
numerical_cols=['Age','Fare']



categorical_cols=['Pclass','Sex','Embarked','Family']



numerical_transformer = SimpleImputer(strategy='median')



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder())

])





preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



model = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', RandomForestClassifier(random_state=0, 

                                                               n_estimators=600, max_depth=5))

                             ])

model.fit(X,y);
X_test=test[features]

X.head()
preds=model.predict(X_test)
from sklearn.model_selection import cross_val_score

scores=cross_val_score(model,X,y)

#print(scores)

print('Average: {}'.format(np.mean(scores)))
result = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds}) #saving our results

result.to_csv('submission.csv', index=False)