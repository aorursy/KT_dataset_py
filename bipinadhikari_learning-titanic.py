# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Viz import
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

# Scikit learn for model building
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data = train_data.set_index('PassengerId')
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data = test_data.set_index('PassengerId')
test_data.head()
# Lets add a different category of child as well to male and female
# If the age<16 , lets return child .Otherwise return male or female as it is.

def check_child(age):
    
    if age <= 16:
        return 1
    elif age <= 32 and age > 16:
        return 2
    elif age <= 48 and age > 32:
        return 3
    elif age <= 64 and age > 48:
        return 4
    return 5
train_data['person'] = train_data['Age'].apply(check_child)
test_data['person'] = test_data['Age'].apply(check_child)
train_data.head()
test_data.head()
# Gender plot
sns.catplot('Sex', data= train_data, kind= 'count')
sns.catplot('Pclass', data= train_data, kind= 'count')
sns.catplot('Age', data=train_data, kind='count')
sns.catplot('Pclass', data= train_data, hue= 'Sex', kind= 'count')
# Lets plot male ,female and child

sns.catplot('Pclass', data= train_data, hue= 'person', kind= 'count')
# Distribution plot

as_fig = sns.FacetGrid(train_data, hue= 'Pclass', aspect=5)
as_fig.map(sns.kdeplot, 'Age', shade= True)

oldest= train_data['Age'].max()
as_fig.set(xlim= (0, oldest))
as_fig.add_legend()

as_fig = sns.FacetGrid(train_data, hue= 'Sex', aspect=5)
as_fig.map(sns.kdeplot, 'Age', shade= True)

oldest= train_data['Age'].max()
as_fig.set(xlim= (0, oldest))
as_fig.add_legend()
sns.catplot('Embarked', data= train_data, kind= 'count')
sns.catplot('Embarked', data= train_data, hue= 'Pclass', kind ='count')
sns.catplot('Embarked', data= train_data, hue= 'Sex', kind ='count')
corr = train_data.corr()

corr
plt.figure(figsize= (10,10))
sns.heatmap(corr, vmax=0.8, linewidths=0.01,
           square=True, annot= True, cmap= 'YlGnBu', linecolor= 'white')
plt.title('Correlation between features')
train_data['Survived'].value_counts()
len(train_data)
train_data.hist(column= 'Survived', by= ['Sex'], sharey= True);
pd.crosstab(train_data['Survived'], train_data['Sex'], margins= True)
train_data[['Sex','Survived']].groupby('Sex').mean()
train_data['Age'].value_counts()
train_data.info()
test_data.info()
train_data['Embarked'].unique()
train_data['Embarked'].value_counts()
pd.crosstab(train_data['Survived'],train_data['Embarked'],margins=True)
pd.crosstab(train_data['Survived'],train_data['Pclass'],margins=True)
train_data[['Survived', 'Pclass']].groupby('Pclass').mean()
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', height=2, aspect=1.4)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend();
train_data[['Sex', 'Pclass', 'Age']].groupby(['Sex', 'Pclass']).median()
train_data[['Sex', 'Pclass', 'Age']].groupby(['Sex', 'Pclass']).median().to_dict()['Age']
class AgeImputer(BaseEstimator, TransformerMixin):
    # input is a Dataframe
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.ages = train_data[['Sex', 'Pclass', 'Age']].groupby(['Sex', 'Pclass']).median().to_dict()['Age']
        return self
    
    def transform(self, X):
        X = X.copy()
        for gender in ['female', 'male']:
            for pclass in range(1,4):
                X.loc[(X['Age'].isnull()) 
                     & (X['Sex'] == gender)
                     & (X['Pclass'] == pclass), 'Age'] = self.ages[(gender, pclass)]
                
        return X
        
age_imputer = AgeImputer()
new_train_data = age_imputer.fit_transform(train_data)
new_train_data.info() # missing age values are replaced in the new train data using the age imputer
new_train_data['Survived'].value_counts()
new_train_data.info()
train_data['Age'].isnull().value_counts()
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_train.info()
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

features = Pipeline([
    ('age', AgeImputer()),
    ('impute', ColumnTransformer([
        ('fare', SimpleImputer(strategy= 'median'), ['Fare']),
        ('embarked', SimpleImputer(strategy= 'most_frequent'), ['Embarked']),
        ('keep', 'passthrough', ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
    ])),
     ('onehot', ColumnTransformer([
            ('code', OneHotEncoder(), [1,2,3])
        ], remainder= 'passthrough'))
])
features.fit_transform(X_train)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

est = Pipeline([
    ('features', features),
    ('rf', GridSearchCV(RandomForestClassifier(), 
                        param_grid = {'max_depth': [10,15,20,25,35],'min_samples_leaf': [5,20]},
                       cv= 5, verbose=1))
])
est.fit(X_train, y_train)
est.score(X_train, y_train)
est.predict(test_data)
est.named_steps['rf'].best_params_
rf_output = pd.DataFrame(est.predict(test_data), index= test_data.index, columns= ['Survived'])
rf_output.head()
rf_output.to_csv('/kaggle/working/titanic_final_rf.csv')
