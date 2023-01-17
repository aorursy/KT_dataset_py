# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

data.head(15)
data.info()
sb.countplot('Survived', data=data)

dummy = data['Survived'].value_counts()

print(dummy)

print('Survival rate :', dummy[1]/sum(dummy)*100,'%')

plt.show()
plt.figure(figsize=(10,5))

plt.subplot(121)

sb.countplot('Sex',data=data)

plt.subplot(122)

sb.countplot('Sex',hue='Survived',data=data,)

data['Sex'].value_counts()

plt.show()
plt.figure(figsize=(20,5))

plt.subplot(131)

sb.countplot('Pclass',data=data)

plt.subplot(132)

sb.countplot('Pclass',hue='Survived',data=data,)

plt.subplot(133)

sb.countplot('Survived',hue='Pclass',data=data,)

data['Pclass'].value_counts()

plt.show()
pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True).style.background_gradient(cmap='summer_r')
sb.catplot('Pclass', 'Survived', hue='Sex', data=data, kind='point')

plt.show()
plt.figure(figsize=(10,5))

plt.subplot(121)

sb.countplot('Embarked', data=data)

plt.subplot(122)

sb.countplot('Embarked',hue='Survived',data=data,)

data['Embarked'].value_counts()

plt.show()
sb.countplot('Embarked',hue='Pclass',data=data,)

plt.show()
from statistics import median,mean

dummy = data['Age']*data['Survived']

dummy = [i for i in dummy if i!=0 and str(i) != 'nan']

print('Oldest person to survive:',max(dummy),'years')

print('Youngest survivor:',min(dummy),'years')

print('Average age of the survivors:',mean(dummy),'years')

print('Median age of the survivors:',median(dummy),'years')
dummy = data.loc[(data.Survived == 0),'Age']

dummy = [i for i in dummy if i!=0 and str(i) != 'nan']

print('Non-survivors')

print('Oldest person to die:',max(dummy),'years')

print('Youngest non-survivor:',min(dummy),'years')

print('Average age of the non-survivours:',mean(dummy),'years')

print('Median age of the non-survivours:',median(dummy),'years')
dummy = data.loc[(data.Survived == 0) & (data.Sex=='male') & (data.Pclass == 1),'Age']

dummy = [i for i in dummy if i!=0 and str(i) != 'nan']

print('Male data who survived')

print('Oldest person to survive:',max(dummy),'years')

print('Youngest survivor:',min(dummy),'years')

print('Average age of the survivours:',mean(dummy),'years')

print('Median age of the survivours:',median(dummy),'years')
dummy = data.loc[(data.Survived == 1) & (data.Sex=='female') & (data.Pclass == 2),'Age']

dummy = [i for i in dummy if i!=0 and str(i) != 'nan']

print('Female data who survived')

print('Oldest person to survive:',max(dummy),'years')

print('Youngest survivor:',min(dummy),'years')

print('Average age of the survivours:',mean(dummy),'years')

print('Median age of the survivours:',median(dummy),'years')
dummy = data.loc[(data.Survived == 0) & (data.Sex=='male') & (data.Pclass == 2),'Age']

dummy = [i for i in dummy if i!=0 and str(i) != 'nan']

print('Male data who survived')

print('Oldest person to survive:',max(dummy),'years')

print('Youngest survivor:',min(dummy),'years')

print('Average age of the survivours:',mean(dummy),'years')

print('Median age of the survivours:',median(dummy),'years')
if data.Cabin.isnull().any():

    data.loc[~(data.Cabin.isnull()), 'Cabin'] = 1

    data.loc[(data.Cabin.isnull()), 'Cabin'] = 0

    test_data.loc[~(data.Cabin.isnull()), 'Cabin'] = 1

    test_data.loc[(data.Cabin.isnull()), 'Cabin'] = 0

data.head()
sb.countplot('Cabin',hue='Survived',data=data,)

plt.show()
def include_initial(data=data):

    data['Initial']=0

    for i in data:

        data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')

    data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',

                               'Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss',

                                'Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

def impute_age(data=data):

    data.loc[(data.Age.isnull()) & (data.Initial=='Mr'),'Age']=33

    data.loc[(data.Age.isnull()) & (data.Initial=='Mrs'),'Age']=36

    data.loc[(data.Age.isnull()) & (data.Initial=='Master'),'Age']=5

    data.loc[(data.Age.isnull()) & (data.Initial=='Miss'),'Age']=22

    data.loc[(data.Age.isnull()) & (data.Initial=='Other'),'Age']=46
include_initial(data)

include_initial(test_data)

impute_age(data)

impute_age(test_data)
print(data.groupby('Initial')['Age'].mean())
display(pd.crosstab(data.Parch,data.Survived).style.background_gradient('summer_r'))

display((pd.crosstab(data.Parch, data.Survived).apply(lambda row:row*100/row.sum(),axis=1).style.background_gradient('summer_r')))

# data['Parch'].value_counts()
from IPython.display import display, HTML



CSS = """

.output {

    flex-direction: row;

}

"""



HTML('<style>{}</style>'.format(CSS))
display(pd.crosstab(data.SibSp,data.Survived).style.background_gradient('summer_r'))

display((pd.crosstab([data.SibSp], data.Survived).apply(lambda row:row*100/row.sum(),axis=1).style.background_gradient('summer_r')))

display(pd.crosstab([data.SibSp,data.Pclass],data.Survived).style.background_gradient('summer_r'))
data['Fsize'] = 0

data['Fsize'] = 0

for i in data:

    data['Fsize'] = data.Parch + data.SibSp

for i in test_data:

    test_data['Fsize'] = test_data.Parch + test_data.SibSp

display(pd.crosstab(data.Fsize,data.Survived).style.background_gradient('summer_r'))

display((pd.crosstab(data.Fsize,data.Survived).apply(lambda r:r*100/r.sum())).style.background_gradient('summer_r'))

plt.figure(figsize=(10,5))

sb.barplot(x='Fsize', y='Survived', data=data)

plt.show()
def FScat(data=data):

    data['FScat'] = 0

    data.loc[(data.Fsize==0),'FScat'] = 'solo'

    data.loc[(data.Fsize>0) & (data.Fsize<4),'FScat'] = 'small'

    data.loc[(data.Fsize>=4),'FScat'] = 'big'

FScat(data)

FScat(test_data)

display(pd.crosstab(data.FScat,data.Survived).style.background_gradient('summer_r'))

plt.figure(figsize=(8,3))

sb.countplot('FScat',hue='Survived',data=data,)

plt.show()
data.head()
y = data['Survived']

features = ['Pclass','Age', 'Fare', 'Cabin', 'Embarked', 'Initial', 'Fsize', 'FScat', 'Survived']

X = data[features]

test_x = test_data[['Pclass','Age', 'Fare', 'Cabin', 'Embarked', 'Initial', 'Fsize', 'FScat']]

X.head()
X.info()
%matplotlib inline

X.hist(bins = 50, figsize=(10,10))

plt.show()
corr_matrix = X.corr()

corr_matrix['Survived'].sort_values(ascending=False)
X['age_cat'] = pd.cut(X['Age'], bins=[0., 10, 25, 45, 60, np.inf], labels = [1,2,3,4,5])

test_x['age_cat'] = pd.cut(test_x['Age'], bins=[0., 10, 25, 45, 60, np.inf], labels = [1,2,3,4,5])
X.head()
# from sklearn.model_selection import StratifiedShuffleSplit

# split = StratifiedShuffleSplit(n_splits = 1,test_size = 0.2, random_state = 42)

# for train_index,test_index in split.split(X, X['age_cat']):

#     train_set = X.loc[train_index]

#     test_set = X.loc[test_index]
# test_set['age_cat'].value_counts()/len(test_set)
# for set_ in (train_set, test_set):

#     set_.drop('age_cat', axis=1, inplace=True)
#comment 

train_set = X.copy()
y_train = train_set['Survived']

# y_test = test_set['Survived']

train_set.drop('Survived', axis=1, inplace=True)

# X_test = test_set.drop('Survived', axis=1, inplace=True)
X_train = train_set.copy()

# X_test = test_set.copy()

X_train.head()
numerical_cols = ['Fare']

categorical_cols = ['Pclass', 'Embarked', 'Cabin', 'Initial', 'FScat', 'age_cat']



# Preprocessing for numerical data

from sklearn.impute import SimpleImputer

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical data

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

from sklearn.compose import ColumnTransformer



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])

# Bundle preprocessing and modeling code 

from sklearn.ensemble import RandomForestClassifier

titanic_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                                   ('model', RandomForestClassifier(random_state=0, n_estimators=500, max_depth=5))

                                  ])



# Preprocessing of training data, fit model 

from sklearn.model_selection import cross_val_score

titanic_pipeline.fit(X_train,y_train)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline, X_train, y_train, cv=10).mean()))
X_test_final = test_x

X_test_final.head()
# Preprocessing of test data, get predictions

predictions = titanic_pipeline.predict(X_test_final)
test_data = pd.read_csv('../input/titanic/test.csv')
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print('Your submission was successfully saved!')