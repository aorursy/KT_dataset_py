# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv') 

train.head(4)
train.shape
train.describe()
train.describe(include=['O'])
train.info()
train.isnull().sum()
test.isnull().sum()
survived = train[train['Survived'] == 1]

not_survived = train[train['Survived'] == 0]



print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))

print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))

print ("Total: %i"%len(train))
train.Pclass.value_counts()

train.groupby('Pclass').Survived.value_counts()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.barplot(x='Pclass', y='Survived', data=train)
train.groupby('Sex').Survived.value_counts()

sns.barplot(x='Sex', y='Survived', data=train)
tab = pd.crosstab(train['Pclass'], train['Sex'])

print (tab)



tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Pclass')

plt.ylabel('Percentage')
sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)
fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(131)

ax2 = fig.add_subplot(132)

ax3 = fig.add_subplot(133)



sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True, ax=ax1)

sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True, ax=ax2)

sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, ax=ax3)
total_survived = train[train['Survived']==1]

total_not_survived = train[train['Survived']==0]

male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]

female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]

male_not_survived = train[(train['Survived']==0) & (train['Sex']=="male")]

female_not_survived = train[(train['Survived']==0) & (train['Sex']=="female")]



plt.figure(figsize=[15,5])

plt.subplot(111)

sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')

sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Age')



plt.figure(figsize=[15,5])



plt.subplot(121)

sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')

sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Female Age')



plt.subplot(122)

sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')

sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Male Age')
plt.figure(figsize=(15,6))

sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)
train.head(4)
def substrings_in_string(big_string, substrings):

    for substring in substrings:

        #print (substring,big_string)

        if not pd.isnull(big_string) and big_string.find( substring) == 0:

            return substring

    #print (big_string)

    return np.nan



train['Family_Size']=train['SibSp']+train['Parch']

train['Fare_Per_Person']=train['Fare']/(train['Family_Size']+1)

#Turning cabin number into Deck

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

train['Deck']=train['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
import matplotlib.pyplot as plt

from xgboost.sklearn import XGBClassifier

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder



X =  train[['Pclass', 'Sex', 'Age', 'Embarked','Fare','Age','Family_Size','Deck' ]]

Y = train['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle= True)



import sklearn as sk

from sklearn.ensemble import RandomForestClassifier



numeric_features = ['Age', 'Fare','Family_Size']

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])



categorical_features = ['Embarked', 'Sex', 'Pclass','Deck']

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer(

    transformers=[

         ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])



clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=8, random_state=0,

                                                            max_features =5,oob_score=True,n_jobs=-1, verbose=0,

                                                            min_samples_leaf=1,  min_samples_split=5

                                                           ))])

 

clf.fit(X_train, y_train)

print('Score: ', clf.score(X_train, y_train)*100)

y_pred_random_forest_training_set = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print ("Random Forest Accuracy " , accuracy_score(y_pred_random_forest_training_set,y_test)*100 ,"%")



test['Family_Size']=test['SibSp']+test['Parch']

test['Fare_Per_Person']=test['Fare']/(test['Family_Size']+1)

#Turning cabin number into Deck

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

test['Deck']=test['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))



X_test =  test[['Pclass', 'Sex', 'Age', 'Embarked','Fare','Age','Family_Size','Deck']]

y_pred_random_forest = clf.predict(X_test)

#submission = pd.DataFrame({

#        "PassengerId": test["PassengerId"],

#        "Survived": y_pred_random_forest

#    })
clf_xg = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', XGBClassifier(learning_rate=0.003,n_estimators=1900,max_depth=6, min_child_weight=0,

                                gamma=0, subs1ample=0.7,

                                colsample_bytree=0.7,

                                scale_pos_weight=1, seed=20,

                                reg_alpha=0.00005))]) 



clf_xg.fit(X_train, y_train)

print('Score: ', clf_xg.score(X_train, y_train)*100)

y_xboost = clf_xg.predict(X_test)



from sklearn.metrics import accuracy_score

print ("Accuracy XGB" , accuracy_score(y_xboost,y_test)*100 ,"%")
y_xgboost = clf_xg.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_xgboost

    })
from sklearn.metrics import confusion_matrix

import itertools



y_pred_random_forest_training_set = clf.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pred_random_forest_training_set)

np.set_printoptions(precision=2)



print ('Confusion Matrix in Numbers')

print (cnf_matrix)

print ('')

true_class_names = ['True Survived', 'True Not Survived']

predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix, 

                             index = true_class_names,

                             columns = predicted_class_names)

sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

 





y_pred_xgb = clf_xg.predict(X_test)

cnf_matrix_cgb = confusion_matrix(y_test, y_pred_xgb)

np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers XGB' )

print (cnf_matrix_cgb)

print ('')

true_class_names = ['True Survived', 'True Not Survived']

predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix_cgb, 

                             index = true_class_names,

                             columns = predicted_class_names)

sns.heatmap(df_cnf_matrix, annot=True, fmt='d') 
submission.head(2)

submission.to_csv('submission.csv', index=False)