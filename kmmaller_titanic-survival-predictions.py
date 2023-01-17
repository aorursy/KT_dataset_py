# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#input data.  split into features and values.  drop cabin as its values are mostly null.

#also from ticket

data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

y_train = data_train['Survived']

X_train = data_train.drop(['Cabin','Ticket'],axis=1)

X_final_test = data_test.drop(['Cabin','Ticket'],axis=1)

X_train = X_train.set_index("PassengerId")

X_final_test = X_final_test.set_index('PassengerId')

X_train.info()

print('---')

X_final_test.info()
#some exploratory plots to get an idea of what features matter.  

# is appears Pclass matters

sns.countplot(X_train['Pclass'], hue=X_train['Survived'])

sns.countplot(X_train['Sex'], hue=X_train['Survived'])

#unsurprisingly, sex matters


sns.countplot(pd.qcut(X_train['Age'],7), hue=X_train['Survived'])

#divide ages into groups.  it appears the only group where more survived then died are children
sns.countplot(pd.qcut(X_train['Fare'],4), hue=X_train['Survived'])

#do the same thing with the fare which shows one had a better chance of survival if one 

#paud more for their ticket
#fix the missing data. must be done for training and test sets in the same way.

X_train['Embarked']=X_train['Embarked'].fillna('S')

X_train['Age']= X_train['Age'].fillna(X_train['Age'].mean())

X_final_test['Age']= X_final_test['Age'].fillna(X_train['Age'].mean())

X_final_test['Fare']=X_final_test['Fare'].fillna(X_train['Fare'].mean())

X_train.info()

print('--')

X_final_test.info()
#inspired by kaggle user: ZlatanKremonic  lets see if we can do anything with the name before 

#disregarding it as a useless feature.

X_train['Name'].head()
X_train['Name_title']= X_train['Name'].apply(lambda x: x.split(',')[1])

X_train['Name_title']= X_train['Name_title'].apply(lambda x: x.split()[0])

X_train['Name_title'].value_counts()

sns.countplot(X_train['Name_title'], hue=X_train['Survived'])

#from this we can see that misters largely did not survive while mrs., miss and master 

# had a better chance!
X_train['Name_len'] = X_train['Name'].apply(lambda x: x.split())

X_train['Name_len'] = X_train['Name_len'].apply(lambda x: len(x))

X_train['Name_len'].value_counts()
sns.countplot(X_train['Name_len'], hue=X_train['Survived'])

#here we people with shorter names were less likely to survive.  this could be another good

#feature
#now that we have made new features, we need to add them to the test data and get rid of the

#old features in both data sets

X_final_test['Name_len'] = X_final_test['Name'].apply(lambda x: x.split())

X_final_test['Name_len'] = X_final_test['Name_len'].apply(lambda x: len(x))

X_final_test['Name_title']= X_final_test['Name'].apply(lambda x: x.split(',')[1])

X_final_test['Name_title']= X_final_test['Name_title'].apply(lambda x: x.split()[0])

X_train = X_train.drop(['Name'],axis=1)

X_final_test = X_final_test.drop(['Name'],axis=1)

X_train = X_train.drop(['Survived'],axis=1)

X_final_test['Name_title'].value_counts()



#X_train['Name_title'].value_counts()

#for name_title there are columns in the train and will not be in the test

# leading to difference dimension feature matrices.  we need to fix that by only allowing

# name_titles that appear in both data sets

good_cols = ["Name_title_"+i for i in X_train['Name_title'].unique() 

             if i in X_final_test['Name_title'].unique()]

print(good_cols)
#categorical data needs to be replaced with dummy variables Sex and Embarked, and name_title





X_train_sex = pd.get_dummies(X_train['Sex'],prefix='Sex')

X_train_Embarked = pd.get_dummies(X_train['Embarked'],prefix='Embarked')

X_train_Pclass = pd.get_dummies(X_train['Pclass'],prefix='Pclass')

X_train_Name_title = pd.get_dummies(X_train['Name_title'],prefix='Name_title')



X_final_test_sex = pd.get_dummies(X_final_test['Sex'],prefix='Sex')

X_final_test_Embarked = pd.get_dummies(X_final_test['Embarked'],prefix='Embarked')

X_final_test_Pclass = pd.get_dummies(X_final_test['Pclass'],prefix='Pclass')

X_final_test_Name_title = pd.get_dummies(X_final_test['Name_title'],prefix='Name_title')



X_train = pd.concat([X_train,X_train_sex,X_train_Embarked,X_train_Pclass,

                     X_train_Name_title[good_cols]],axis=1)

X_train = X_train.drop(['Sex','Embarked','Pclass','Name_title'],axis=1)

X_final_test = pd.concat([X_final_test,X_final_test_sex,X_final_test_Embarked,

                          X_final_test_Pclass,X_final_test_Name_title[good_cols]],axis=1)

X_final_test = X_final_test.drop(['Sex','Embarked','Pclass','Name_title'],axis=1)

X_train.info()

print('---')

X_final_test.info()
#train with all the features

X_train2,X_test2,y_train2,y_test2 = train_test_split(X_train,y_train,test_size=0.2,random_state=42,stratify=y_train)

pipeline = make_pipeline(preprocessing.StandardScaler(),

                        RandomForestClassifier(n_estimators=700, oob_score=True))

hyperparameters = {'randomforestclassifier__min_samples_leaf':[1,5,10],

                  'randomforestclassifier__max_depth':[None,10,7,5],

                  'randomforestclassifier__min_samples_split':[2,4,8,12]}

grid_search = GridSearchCV(pipeline,hyperparameters,cv=10)

grid_search.fit(X_train2,y_train2)

print(grid_search.best_params_)

grid_search.refit

pred = grid_search.predict(X_test2)

print("accuracy score: ", accuracy_score(y_test2,pred))
rf=RandomForestClassifier(n_estimators=700,oob_score=True,max_depth=10,min_samples_leaf=5,

                         min_samples_split=8)

rf.fit(X_train2,y_train2)

pred=rf.predict(X_test2)



pd.concat((pd.DataFrame(X_train2.iloc[:, 1:].columns, columns = ['variable']), 

           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 

          axis = 1).sort_values(by='importance', ascending = False)[:20]
final_pred = rf.predict(X_final_test)

X_final_test=X_final_test.reset_index();

predictions = pd.DataFrame(final_pred,columns=['Survived'])

predictions = pd.concat((X_final_test['PassengerId'],predictions),axis=1)
predictions.shape
predictions.to_csv('submission1.csv', sep=",", index = False)
