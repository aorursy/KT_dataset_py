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
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
sns.set_style('whitegrid')
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
train = df_train.copy()
test = df_test.copy()
print('Train data','\n')
print(train.info(),'\n')
print(train.describe(),'\n')

print('Test data','\n')
print(test.info(),'\n')
print(test.describe(),'\n')

print('Train data : {}'.format(train.shape))
print('Test data : {}'.format(test.shape))
train
test
assert 'PassengerId' in train, 'hi'
def col_counts(col):
    '''Print the value counts of columns from train and test data.
    
    Args:
        col (str) : name of columns of train and test data
    
    Returns:
       Print :
           train's column's value counts
           test column's value counts
    '''
    if col in train.columns:
        print('Train\'s {} : '.format(col))
        print(train[col].value_counts(),'\n')
    else:
        print('Train\'s data does not have {} column.'.format(col))
        
    if col in test.columns:
        print('Test\'s {} : '.format(col))
        print(test[col].value_counts())
    else:
        print('Test\'s data does not have {} column.'.format(col))

        
def drop_col(data, column):
    '''Drop column from data.
    
    Args:
        data (dataframe) : dataframe to drop columns on
        column (list, str) : columns to drop
        
    Return :
        None
    '''
    return data.drop(column, inplace = True,axis = 1)


def title_type(row):
    '''Return str based on the input
    
    Args:
        row (str) : row data that contain the title of the name
        
    Return:
        str : categories for the input 
    '''
    if row in ['Don', 'Mme',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess',
       'Jonkheer','Dona','Dr','Rev']:
        # label as rare for titles that are low in counts
        return 'Rare'
    elif row == 'Miss':
        return 'Ms'
    else:
        return row
    
    
def age_diff(row):
    '''Return the category of age based on input age.
    
    Args:
        row (int) : row data that contain the age
        
    ReturnL
        str : categories of the input age
    '''
    if row < 18:
        return 'Child'
    elif (row < 60) & (row >=18):
        return 'Adult'
    else:
        return 'Elderly'
drop_col(train, ['PassengerId', 'Ticket'])
drop_col(test, ['PassengerId', 'Ticket'])
col_counts('Survived')
# split the name string based on ',' first and then '.'
train['Title'] = train.Name.apply(lambda x : x.split(',')[1].split('.')[0].strip())
test['Title'] = test.Name.apply(lambda x : x.split(',')[1].split('.')[0].strip())
    
train['Title'] = train.Title.apply(title_type)
test['Title'] = test.Title.apply(title_type)

# drop 'Name' column
drop_col(train, 'Name')
drop_col(test,'Name')
col_counts('Title')
# rename the category
train['Sex'] = train.Sex.map({'male':'Male','female':'Female'})
test['Sex'] = test.Sex.map({'male':'Male','female':'Female'})
col_counts('Sex')
# add the number of siblings, spouse, parents, children and the passenger itself
train['Family'] = train.SibSp + train.Parch + 1
test['Family'] = test.SibSp + test.Parch + 1

# drop 'SibSp' and 'Parch' columns to prevent repeatative features
drop_col(train, ['SibSp','Parch'])
drop_col(test, ['SibSp','Parch'])
col_counts('Family')
# Divide the family size into 4 categories
train['Family_type'] = pd.cut(train.Family, [0,1,4,7,11], labels = ['Single', 'Small', 'Medium', 'Large'])
test['Family_type'] = pd.cut(test.Family, [0,1,4,7,11], labels = ['Single', 'Small', 'Medium', 'Large'])
col_counts('Family_type')
# divide the age of passengers into 3 different categories
train['Age_cat'] = train.Age.apply(age_diff)
test['Age_cat'] = test.Age.apply(age_diff)
col_counts('Age_cat')
# Extract the first alphabert from the cabin name
train['Cabin_floor'] = train.Cabin.apply(lambda x: list(str(x))[0])
train['Cabin_floor'] = train.Cabin_floor.replace('n', np.nan)

test['Cabin_floor'] = test.Cabin.apply(lambda x: list(str(x))[0])
test['Cabin_floor'] = test.Cabin_floor.replace('n', np.nan)

# drop 'Cabin' column
drop_col(train,'Cabin')
drop_col(test,'Cabin')
col_counts('Cabin_floor')
# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Validation model performance
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay, roc_auc_score

# Hyperparameter tuning
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score

# Model
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

seed = 225
y = train['Survived']
X = train.drop('Survived',axis = 1)
# define columns for numerical and categorical
num_cols = ['Fare']

cat_cols = ['Pclass', 'Sex','Embarked','Title','Family_type','Age_cat']

# pipeline for preprocessing of numerical and categorical data
cat_transformer = Pipeline(steps = [('Cat_Imputer', SimpleImputer(strategy = 'most_frequent')),('OneHotEncoder',OneHotEncoder(handle_unknown = 'ignore'))])
num_transformer = Pipeline(steps = [('Num_Imputer', SimpleImputer(strategy = 'median'))])

preprocessor = ColumnTransformer(transformers = [('num', num_transformer, num_cols), ('cat',cat_transformer, cat_cols)])

# pipeline for modeling
titanic_pipeline = Pipeline(steps = [('Preprocessor',preprocessor),('XG', XGBClassifier(random_state = seed))])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)

params = {'XG__learning_rate' : [0.1,0.2], 'XG__gamma' : [0.001,0.01,1,10],'XG__max_depth' : [4,6,8,10], 'XG__n_estimators' : [400,500]}

searcher = GridSearchCV(titanic_pipeline,params, cv = 3, verbose = 1, n_jobs = -1 )

searcher.fit(X_train,y_train)

print('Best params : {}'.format(searcher.best_params_))
print('Best score : {:.2f}'.format(searcher.best_score_))

y_pred_train = searcher.predict(X_train)
y_pred_test = searcher.predict(X_test)

print('XGBoost\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))
print('XGBoost\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test,y_pred_test))
print('XGBoost\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))
# define columns for numerical and categorical
num_cols = ['Fare','Age']

cat_cols = ['Pclass', 'Sex','Embarked','Title','Family_type']

# pipeline for preprocessing of numerical and categorical data
cat_transformer = Pipeline(steps = [('Cat_Imputer', SimpleImputer(strategy = 'most_frequent')),('OneHotEncoder',OneHotEncoder(handle_unknown = 'ignore'))])
num_transformer = Pipeline(steps = [('Num_Imputer', SimpleImputer(strategy = 'median'))])

preprocessor = ColumnTransformer(transformers = [('num', num_transformer, num_cols), ('cat',cat_transformer, cat_cols)])

# pipeline for modeling
titanic_pipeline = Pipeline(steps = [('Preprocessor',preprocessor),('XG', XGBClassifier(random_state = seed, learning_rate = 0.1))])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)

params = { 'XG__gamma' : [0.001,0.01,1,10,100,1000],'XG__max_depth' : [2,4,6,8,10], 'XG__n_estimators' : [400,500]}

searcher_xg = GridSearchCV(titanic_pipeline,params, cv = 3, verbose = 1, n_jobs = -1 )

searcher_xg.fit(X_train,y_train)

print('Best params : {}'.format(searcher_xg.best_params_))
print('Best score : {:.2f}'.format(searcher_xg.best_score_))

y_pred_train = searcher_xg.predict(X_train)
y_pred_test = searcher_xg.predict(X_test)

print('XGBoost\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))
print('XGBoost\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test,y_pred_test))
print('XGBoost\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))
num_cols = ['Fare']

cat_cols = ['Pclass', 'Sex','Embarked','Title','Family_type','Age_cat']

cat_transformer = Pipeline(steps = [('Cat_Imputer', SimpleImputer(strategy = 'most_frequent')),('OneHotEncoder',OneHotEncoder(handle_unknown = 'ignore'))])
num_transformer = Pipeline(steps = [('Num_Imputer', SimpleImputer(strategy = 'median')), ('Scaler', RobustScaler())])

preprocessor = ColumnTransformer(transformers = [('num', num_transformer, num_cols), ('cat',cat_transformer, cat_cols)])

titanic_pipeline = Pipeline(steps = [('Preprocessor',preprocessor),('SVC', SVC(random_state = seed))])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = seed)

parameters = {'SVC__C':[0.1, 1, 10,100], 'SVC__gamma':[ 0.001, 0.01, 0.1,1,10]}
searcher = GridSearchCV(titanic_pipeline, parameters, cv = 5, n_jobs = -1, verbose = 1)

searcher.fit(X_train,y_train)

print('Best params : {}'.format(searcher.best_params_))
print('Best score : {:.2f}'.format(searcher.best_score_))

y_pred_train = searcher.predict(X_train)
y_pred_test = searcher.predict(X_test)

print('SVC\'s train score : {:.3f}'.format(accuracy_score(y_train,y_pred_train)))
print('SVC\'s test score : {:.3f}'.format(accuracy_score(y_test,y_pred_test)))
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test,y_pred_test))
print('SVC\'s roc score : {:.3f}'.format(roc_auc_score(y_test,y_pred_test)))
output = searcher_xg.predict(test)
df_test['Survived'] = output
df_test = df_test[['PassengerId', 'Survived']]
print(df_test.shape)
df_test.to_csv('submission_2.csv', index=False)