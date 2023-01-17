# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def survival_rate(groups):

    '''

    calculate a survival rate by the groups specified

    groups: a list or string, columns to group by

    '''

    

    output = data.groupby(groups).agg({'Survived':['sum','count']})

    output['survival_rate'] = output.iloc[:,0]/output.iloc[:,1]

    return output
data = pd.read_csv('/kaggle/input/titanic/train.csv')

data.head()
data.info()
data.describe()
fig, ax =plt.subplots(2,3,figsize=(15,10))

sns.countplot(data['Survived'], ax=ax[0,0])

sns.countplot(data['Sex'], ax=ax[1,0])

sns.countplot(data['Pclass'], ax=ax[0,1])

sns.countplot(data['SibSp'], ax=ax[1,1])

sns.countplot(data['Parch'], ax=ax[0,2])

sns.countplot(data['Embarked'], ax=ax[1,2])



fig.show()
sns.catplot(x='Survived',kind='count',data=data)
sns.catplot(x='Survived',kind='count',data=data,col='Sex')
sns.catplot(x='Survived',kind='count',data=data,col='Sex',row='Pclass')
# Look at the survival rate by sex & class

survival_rate(['Sex','Pclass'])
g = sns.catplot(x='Age',y='Survived',data=data,kind='violin',orient='h')

g.set(xticks = np.arange(-10,100,10))
g = sns.catplot(x='Age',y='Pclass',data=data,kind='boxen',orient='h', hue='Survived')

g.set(xticks = np.arange(0,100,5))
# those under 15 - There are only 14 rows and all are 1 or under, however, most have survived

data[data['Age'] <= 15 & data['Pclass'].isin([1,2,3])]
data['Child'] = (data['Age'] <= 15)*1
g = sns.catplot(y='Fare',x='Survived',data=data,kind='boxen')

axes = g.axes

# axes[0,0].set_ylim(0,300)
g = sns.catplot(y='Fare',x='Pclass',data=data,kind='boxen',hue='Survived')

axes = g.axes

axes[0,0].set_ylim(0,300)
sns.catplot(x='Parch',data=data,kind='count',col='Survived',row='Sex')
data['Parch_0'] = (data['Parch'] == 0)*1
survival_rate('Parch_0')
sns.catplot(x='SibSp',data=data,kind='count',col='Survived')
survival_rate('SibSp')
sns.catplot(x='SibSp',data=data,kind='count',col='Pclass')
data['SibSp_0'] = (data['SibSp'] == 0)*1
survival_rate('SibSp_0')
sum(data['Cabin'].isnull())/data.shape[0]
data[~data['Cabin'].isnull()][['Name','Cabin']].tail(10)
# generate a cabin deck column

data['CabinDeck'] = data['Cabin'].str.extract('([ABCDEFG])[0-9]{1,3}')
data[['Name','Cabin','CabinDeck']][~data['Cabin'].isnull() & data['CabinDeck'].isnull()]
sns.countplot(x='CabinDeck',data=data,order=['A','B','C','D','E','F','G'],hue='Survived')
survival_rate('CabinDeck')
data['Name']
data['Title'] = data['Name'].str.extract('([A-Za-z]*)\.')
sns.catplot(x='Title',data=data,kind='count',hue='Survived')

plt.xticks(rotation=90)
survival_rate('Title').sort_values(by=('Survived','count'),ascending=False)
sns.catplot(y='Age',x='Title',data=data,kind='boxen')

plt.xticks(rotation=90)
sns.catplot(x='Title',data=data,kind='count',hue='Pclass')

plt.xticks(rotation=90)
sns.catplot(x='Embarked',data=data,kind='count',hue='Survived')
survival_rate('Embarked')
sns.catplot(x='Embarked',data=data,hue='Pclass',kind='count')
data.describe()
from sklearn.preprocessing import StandardScaler



standard = StandardScaler()

pd.DataFrame(standard.fit_transform(data[['Age','Fare']]),columns = ['Age','Fare']).describe()

# data.dtypes.values != object]].drop('PassengerId',axis=1
sns.heatmap(data.corr())
X = data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Child','Parch_0','SibSp_0','CabinDeck','Title']]



y = data['Survived']
# convert Pclass to an integer

X['Pclass'] = X['Pclass'].astype(str)



# # assign categorical features - objects or Pclass

# X = pd.get_dummies(X,drop_first=True)
sns.heatmap(X.corr())
# # Creating a pipeline

# # We can add a function to the pipeline by using FeatureTransformer. It was inconvenient as I was adding columns on the numpy array, which doesn't have column names, hence I could risk adding feature off of the wrong column

# def new_features(Y):

#     Z = Y.copy()

#     A = np.zeros((X.shape[0],Z.shape[1]+2))

#     A[:1] = (Z[:,X.columns.get_loc("Parch")] == 1)*1

#     return A



# Pipeline('ft',FunctionTransformer(new_features))

    
from sklearn.metrics import accuracy_score

accuracy_score(y,(X.Sex == 'female')*1)
def model_udf(estimator_name,estimator,parameters,X,y):

    '''

    Return a fitted model on the X and y

    estimator name : string for the name of the estimator step in the pipeline

    estimator : sklearn estimator e.g. KNeighborsClassifier

    parameters : dictionary, parameter grid for Grid Search

    '''

#     steps = [('imputation', SimpleImputer(missing_values=np.nan, strategy='median')),

#              ('scaler',StandardScaler()),

#              ('onehot', OneHotEncoder(handle_unknown='ignore')),

#             (estimator_name, estimator)]



#     pipe = Pipeline(steps)



    numeric_features = list(X._get_numeric_data().columns)

    numeric_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='median')),

        ('scaler', StandardScaler())])



    categorical_features = list(set(X.columns) - set(X._get_numeric_data().columns))

    categorical_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

        ('onehot', OneHotEncoder(handle_unknown='ignore'))])



    preprocessor = ColumnTransformer(

        transformers=[

            ('num', numeric_transformer, numeric_features),

            ('cat', categorical_transformer, categorical_features)])



    # Append classifier to preprocessing pipeline.

    # Now we have a full prediction pipeline.

    pipe = Pipeline(steps=[('preprocessor', preprocessor),

                          (estimator_name, estimator)])





    parameters = parameters



    clf = GridSearchCV(pipe, param_grid=parameters,return_train_score=True)



    clf.fit(X,y)

    

    return clf

numeric_features = list(X._get_numeric_data().columns)

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])



categorical_features = list(set(X.columns) - set(X._get_numeric_data().columns))

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])



# Append classifier to preprocessing pipeline.

# Now we have a full prediction pipeline.

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', KNeighborsClassifier())])

clf = model_udf(

    'knn'

    ,KNeighborsClassifier()

    ,parameters = {'knn__n_neighbors':np.arange(1,31)},X=X

    ,y=y

)
f'The Grid Search best score is {clf.best_score_:0.1%} and the best parameters are {clf.best_params_}'
cv_results = pd.DataFrame(clf.cv_results_)[['param_knn__n_neighbors','params','mean_score_time','mean_train_score','mean_test_score']]
cv_results.head()
fig, ax = plt.subplots()

ax.plot(cv_results['param_knn__n_neighbors'],cv_results['mean_test_score'],label='mean_test_score')

ax.plot(cv_results['param_knn__n_neighbors'],cv_results['mean_train_score'],label='mean_train_score')

ax.legend()
from sklearn.linear_model import SGDClassifier

clf = model_udf(

    'sgd'

    , SGDClassifier()

    ,parameters = {'sgd__alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 

             'sgd__loss':['hinge','log'], 'sgd__penalty':['l1','l2']},X=X

    ,y=y

)
clf.score(X,y)
f'The Grid Search best score is {clf.best_score_:0.1%} and the best parameters are {clf.best_params_}'
# # Needs review since using onehotencoder https://stackoverflow.com/questions/39043326/computing-feature-importance-with-onehotencoded-features

# def feature_importance(gridsearch_object,estimator_step):

#     plt.figure(figsize=(10,10))

#     sns.barplot(x='b',y='a',data=pd.DataFrame({'a':X.columns,'b':clf.best_estimator_.named_steps['sgd'].coef_[0]}).sort_values('b'))



# feature_importance(clf,'sgd')



# clf.best_estimator_.named_steps['preprocessor'].transformers[1][1].named_steps['onehot'].get_feature_names(categorical_features)
from sklearn.ensemble import RandomForestClassifier





clf = model_udf(

    'rfc'

    , RandomForestClassifier()

    ,parameters = {'rfc__n_estimators':[100,350,500],'rfc__max_features':['log2','auto','sqrt'],'rfc__min_samples_leaf':[2,10,30]}

    ,X=X

    ,y=y

)
f'The Grid Search best score is {clf.best_score_:0.1%} and the best parameters are {clf.best_params_}'
clf.score(X,y)
def data_transform(data):

    data['Title'] = data['Name'].str.extract('([A-Za-z]*)\.')

    data['CabinDeck'] = data['Cabin'].str.extract('([ABCDEFG])[0-9]{1,3}')

    data['SibSp_0'] = (data['SibSp'] == 0)*1

    data['Child'] = (data['Age'] <= 15)*1

    data['Parch_0'] = (data['Parch'] == 0)*1

    data = data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Child','Parch_0','SibSp_0','CabinDeck','Title']]

    # convert Pclass to an integer

    data['Pclass'] = data['Pclass'].astype(str)

    return data
X_test = pd.read_csv('/kaggle/input/titanic/test.csv')

X_test = data_transform(X_test)
X_test = pd.read_csv('/kaggle/input/titanic/test.csv')

data_transform(X_test)
predictions = clf.predict(X_test)
output = pd.DataFrame({'PassengerId':X_test.PassengerId, 'Survived':predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")