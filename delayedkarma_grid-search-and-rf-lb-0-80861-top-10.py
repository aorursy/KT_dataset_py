# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def feature_engg_train(df):
    
    import pandas as pd
    
    
    df['Title'] = df['Name'].map(lambda name:name.split('.')[0].split(',')[1].strip())
    
    titles_dict_train = {}

    for title in ['Capt','Col','Major']:
        titles_dict_train[title] = 'Officer'
    for title in ['Rev','Dr']:
        titles_dict_train[title] = 'Other'
    for title in ['Don','the Countess','Jonkheer','Lady','Sir']:
        titles_dict_train[title] = 'Royalty'
    for title in ['Ms','Miss','Mlle']:
        titles_dict_train[title] = 'Miss'
    for title in ['Mrs','Mme']:
        titles_dict_train[title] = 'Mrs'
    titles_dict_train['Mr'] = 'Mr'
    titles_dict_train['Master'] = 'Master'
    
    df['Title'] = df['Title'].map(titles_dict_train)
    
    title_dummies = pd.get_dummies(df['Title'],prefix='Title')
    df = pd.concat([df,title_dummies],axis=1)
    
    grouped_sex_class = df.groupby(['Sex','Pclass','Title'])
    
    def impute_median(series):
        return(series.fillna(series.median()))
    
    df['Age'] = grouped_sex_class['Age'].transform(impute_median)
    
    df['Fare'].fillna(df['Fare'].mean(),inplace=True)
    
    df['Embarked'].fillna('S',inplace=True)
    
    embarked_dummies = pd.get_dummies(df['Embarked'],prefix='Embarked')
    df = pd.concat([df,embarked_dummies],axis=1)
    
    df['Cabin'].fillna('X',inplace=True)
    df['Cabin_clean'] = df['Cabin'].apply(lambda x: x[0])
    
    cabin_dummies = pd.get_dummies(df['Cabin_clean'],prefix='Cabin')
    df = pd.concat([df,cabin_dummies],axis=1)
    
    sex_dict = dict(zip(df['Sex'].unique(),range(0,2)))
    df['Sex'] = df['Sex'].map(sex_dict)
    
    pclass_dummies = pd.get_dummies(df['Pclass'],prefix='Pclass')
    df = pd.concat([df,pclass_dummies],axis=1)
    
    df['Family'] = df['SibSp'] + df['Parch'] + 1 #Including passenger
    df['Family_single'] = df['Family'].map(lambda x: 1 if x==1 else 0)
    df['Family_small'] = df['Family'].map(lambda x: 1 if 2<=x<=4 else 0)
    df['Family_large'] = df['Family'].map(lambda x: 1 if x>=4 else 0)
    
    df.drop(['Name','Pclass','Ticket','Cabin','Cabin_clean',\
              'Embarked','Title'],axis=1,inplace=True)
  


    return(df)

def feature_engg_test(df):
    
    import pandas as pd

    df['Title'] = df['Name'].map(lambda name:name.split('.')[0].split(',')[1].strip())
    
    titles_dict_test = { 'Col': 'Officer',
                         'Dona': 'Royalty','Dr': 'Other', 'Master': 'Master', 'Miss': 'Miss',\
                        'Mr': 'Mr', 'Mrs': 'Mrs', 'Ms': 'Miss', 'Rev': 'Other'}
    
    df['Title'] = df['Title'].map(titles_dict_test)
    test_dummies_title = pd.get_dummies(df['Title'],prefix='Title')
    df = pd.concat([df,test_dummies_title],axis=1)
    
    df['Fare'].fillna(df['Fare'].mean(),inplace=True)
    
    grouped_sex_class_test = df.groupby(['Sex','Pclass','Title'])

    def impute_median(series):
        return series.fillna(series.median())

    df['Age'] = grouped_sex_class_test['Age'].transform(impute_median)
    
    
    df['Embarked'].fillna('S',inplace=True)
    
    test_dummies_embarked = pd.get_dummies(df['Embarked'],prefix='Embarked')
    df = pd.concat([df,test_dummies_embarked],axis=1)
    
    df['Cabin'].fillna('X',inplace=True)
    df['Cabin_clean'] = df['Cabin'].apply(lambda x:x[0])
    
    test_dummies_cabin = pd.get_dummies(df['Cabin_clean'],prefix='Cabin')
    df = pd.concat([df,test_dummies_cabin],axis=1)
    
    sex_dict = dict(zip(df['Sex'].unique(),range(0,2)))
    df['Sex'] = df['Sex'].map(sex_dict)
    
    test_dummies_pclass = pd.get_dummies(df['Pclass'],prefix='Pclass')
    df = pd.concat([df,test_dummies_pclass],axis=1)
    
    df['Family'] = df['Parch'] + df['SibSp'] + 1
    df['Family_single'] = df['Family'].map(lambda x: 1 if x==1 else 0)
    df['Family_small'] = df['Family'].map(lambda x: 1 if 2<=x<4 else 0)
    df['Family_large'] = df['Family'].map(lambda x: 1 if x>=4 else 0)
    
    df.drop(['Name','Pclass','Ticket','Cabin','Cabin_clean',\
              'Embarked','Title'],axis=1,inplace=True)
    
    return(df)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set_style('white')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train = feature_engg_train(df_train)
df_test = feature_engg_test(df_test)

features_train = df_train.drop(['Survived','Cabin_T'],axis=1)
target_train = df_train['Survived']

features_test = df_test
print('Number of features in training set are {} and features in test set are {}'.\
      format(features_train.shape[1],features_test.shape[1]))
clf=RandomForestClassifier(n_estimators=100,random_state=42)
clf=clf.fit(features_train,target_train)
features = pd.DataFrame()
features['Feature'] = features_train.columns
features['Importance'] = clf.feature_importances_
features.sort_values(by=['Importance'],ascending=True,inplace=True)
features.set_index('Feature',inplace=True)
features.plot(kind='barh',figsize=(20,20),color='b',alpha=0.75);
model = SelectFromModel(clf, prefit=True,threshold='median')
train_reduced = model.transform(features_train)
train_reduced.shape
test_reduced = model.transform(features_test)
test_reduced.shape
X_train,X_test,y_train,y_test = train_test_split(train_reduced,target_train,random_state=42)

param_grid_rf = {'n_estimators':[100,500],'min_samples_split':[2,5],\
                 'max_depth':[5,10],'min_samples_leaf':[5,10]}
grid_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42, oob_score=True, warm_start=True),param_grid=param_grid_rf,cv=10)
grid_rf.fit(X_train,y_train)
print("Best parameters : {}".format(grid_rf.best_params_))
print("Best cross-validation score : {:.2f}".format(grid_rf.best_score_))
clf_rf = grid_rf.best_estimator_
clf_rf.fit(train_reduced,target_train)

target_test = clf_rf.predict(test_reduced)


df_test['Survived'] = target_test
df_test[['PassengerId','Survived']].to_csv('rf-kaggle-submit.csv',index=False) #Kaggle submission
