# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 

# from sklearn.model_selection import KFold

%matplotlib inline



print('Import Successful')
# Importing Data

df_test = pd.read_csv('../input/test.csv')

df_train = pd.read_csv('../input/train.csv')



print('Train Data',df_train.shape)

print('Test Data',df_test.shape)

df_train.head(n=3)
#Splitting up test and train data

# train_x, test_x, train_y, test_y = train_test_split(df_train, df_train['Survived'],test_size = 0.2)



# del train_x['Survived']

# del test_x['Survived']



# train_x



y_true = df_train['Survived']

df_train.drop('Survived',1, inplace=True)

data = df_train.append(df_test)

print(data.shape)

data.head()

data['Name'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

titles = data['Name'].unique()

titles
def check_title(x):

    title = x['Name']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

    

data['Name'] = data.apply(check_title,axis=1)

data.head()
data['Cabin'] = data['Cabin'].apply(lambda x: x[0] if pd.isnull([x]) == False else 'Missing')

data.head()
data['FamilySize'] = data['SibSp'] + data['Parch']

data['Alone'] = data['FamilySize'].apply(lambda family: 1 if family==0 else 0) 
data['Age']= data.groupby(['Name','Sex','Pclass'])['Age'].transform(lambda age: age.fillna(age.median()))

data['Fare']= data.groupby(['Name','Sex','Pclass'])['Fare'].transform(lambda fare: fare.fillna(fare.median()))

data.head()
data.drop('Ticket',axis=1, inplace=True)

data.head()
data['Sex'] = data['Sex'].apply(lambda sex: 1 if sex=='male' else 0)

data.head()
data.columns
features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Cabin', 'Embarked', 'FamilySize', 'Alone']

df = pd.get_dummies(data[features])

df.head()
df['PassengerId'] = data['PassengerId']

df.set_index('PassengerId', inplace=True)

df.reset_index(inplace=True)

df.head()
train_data = df.ix[:890]

test_data = df.ix[891:]
#Initializing model classifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import fbeta_score, make_scorer

from sklearn.cross_validation import StratifiedKFold

score = make_scorer(fbeta_score, beta=0.5)

#estimator, param_grid, scorer

# cls = RandomForestClassifier()

# parameters = {'max_features':['auto','log2',None],'max_depth': list(range(2,20)), 'n_estimators':list(range(200,300)), 'min_samples_leaf':list(range(1,50)), 'criterion': ['gini','entropy']}

# crossV = StratifiedKFold(y_train, n_folds=5)

# gridSearch = GridSearchCV(estimator=cls, param_grid=parameters, cv= crossV, scoring = score)

# gridSearch.fit(x_train, y_train)
train_data.drop('PassengerId', axis=1, inplace=True)
new_cls = RandomForestClassifier()

new_parameters = {'max_features':[None],'max_depth': [18], 'n_estimators':[239], 'min_samples_leaf':[6], 'criterion': ['gini']}

new_crossV = StratifiedKFold(y_true, n_folds=5)

# new_score = make_scorer(fbeta_score, beta=1)

new_gridSearch = GridSearchCV(estimator=new_cls, param_grid=new_parameters, cv= new_crossV)

new_gridSearch.fit(train_data, y_true)



print('Best score: {}'.format(new_gridSearch.best_score_))

print('Best parameters: {}'.format(new_gridSearch.best_params_))
test_PId =  test_data['PassengerId']

test_data.drop('PassengerId',axis=1,inplace=True)

y_pred = new_gridSearch.predict(test_data)
outputDf = pd.DataFrame()

outputDf['PassengerId'] = test_PId

outputDf['Survived'] = y_pred



#Output it to output.csv

# outputDf.to_csv('../input/output.csv', index=False)

outputDf.head()