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
data_test = pd.read_csv('../input/titanic/test.csv',index_col='PassengerId')

data_train = pd.read_csv('../input/titanic/train.csv',index_col='PassengerId')

data_train
data_train.describe()
data_train.isnull().sum()

#data_test.isnull().sum()
#data_train.Age.fillna(data_train.Age.mean(), inplace=True)

#data_train.Embarked.fillna(1, inplace=True)
for i in data_train.columns:    

    print(i ,': ',len(data_train[i].unique()))

#len(data_train.Name.unique())
columnsForDrop = ['Name', 'Cabin','Ticket','SibSp','Parch']

data_train.drop(columns=columnsForDrop, inplace=True)

################################

data_test.drop(columns=columnsForDrop, inplace=True)



data_train
print(data_train.Sex.value_counts())

print('----------------------------------------------')

print(data_train.Embarked.value_counts())
data_train.Sex.replace('male', 0, inplace=True)

data_train.Sex.replace('female', 1, inplace=True)

data_train.Embarked.replace('S', 1, inplace=True)

data_train.Embarked.replace('C', 2, inplace=True)

data_train.Embarked.replace('Q', 3, inplace=True)

#######################################################

data_test.Sex.replace('male', 0, inplace=True)

data_test.Sex.replace('female', 1, inplace=True)

data_test.Embarked.replace('S', 1, inplace=True)

data_test.Embarked.replace('C', 2, inplace=True)

data_test.Embarked.replace('Q', 3, inplace=True)



data_train

y = data_train.Survived

############################################

X = data_train.drop(columns=['Survived'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.impute import SimpleImputer

# Imputation

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))



#imputed_X_train

#Imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_test.columns = X_test.columns

#imputed_X_train.astype(int)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, f1_score

from sklearn.neighbors import KNeighborsClassifier



from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error ,explained_variance_score, mean_squared_error



'''parameters = {'max_depth':  list(range(6, 100, 10)),

              'max_leaf_nodes': list(range(50, 1001, 10))}



from sklearn.model_selection import GridSearchCV



gsearch = GridSearchCV(estimator=DecisionTreeClassifier(),

                       param_grid = parameters, 

                       scoring='f1',

                       n_jobs=4,cv=5,verbose=7)



gsearch.fit(imputed_X_train, y_train)'''

#########################################################################3

from sklearn.ensemble import RandomForestClassifier



parameters = {'max_depth':  list(range(6, 30, 10)),

              'max_leaf_nodes': list(range(50, 500, 100)),

              'n_estimators': list(range(50, 1001, 150))}



from sklearn.model_selection import GridSearchCV



gsearch = GridSearchCV(estimator=RandomForestClassifier(),

                       param_grid = parameters, 

                       scoring='f1',

                       n_jobs=4,cv=5,verbose=7)



gsearch.fit(imputed_X_train, y_train)
print(gsearch.best_params_.get('max_leaf_nodes'))

print(gsearch.best_params_.get('max_depth'))
#def getMeanError():

    #forest_model = KNeighborsClassifier(n_neighbors=n_est)

   # '''forest_model = DecisionTreeClassifier(max_leaf_nodes=gsearch.best_params_.get('max_leaf_nodes'),

    #                                      max_depth=gsearch.best_params_.get('max_depth'),

    #                                      random_state=1)

    #forest_model.fit(imputed_X_train,y_train)

    #preds = forest_model.predict(imputed_X_test)

    #print(classification_report(y_test,preds))

    #print(accuracy_score(y_test, preds))

    #return f1_score(y_true=y_test, y_pred=preds)'''

#########################################################################

def getMeanError():

    forest_model = RandomForestClassifier(

                         max_depth = gsearch.best_params_.get('max_depth'),

                           max_leaf_nodes = gsearch.best_params_.get('max_leaf_nodes'),

        n_estimators = gsearch.best_params_.get('n_estimators'),random_state=1, n_jobs=4,verbose=7)

    forest_model.fit(imputed_X_train, y_train)

    predictions = forest_model.predict(imputed_X_test)

    #mean_Error = mean_absolute_error(y_true=y_test,y_pred = predictions)

    print(classification_report(y_test,predictions))

    print(accuracy_score(y_test, predictions))

    return f1_score(y_true=y_test, y_pred=predictions)

getMeanError()
##results = {}



##range_Estimation = getMeanError(2)

##for i in range(2,20):

    #getMeanError(i)

  ##  print('-------------------------------------------------------')

    ##results[i] = getMeanError(i)
#range_Estimation = getMeanError(2)

#minEstim = 1

#for i in range(2,20):

    #print(getMeanError(i),'*-*',i)

#    if range_Estimation > getMeanError(i):

 #       minEstim = i

#print(range_Estimation,'>>>',minEstim)

#####19 is the best...'''
##best_max_leaf_nodes = max(results, key=results.get)

##best_max_leaf_nodes
X.describe()
X.Age.fillna(X.Age.mean(), inplace=True)

X.Embarked.fillna(1, inplace=True)

#print(gsearch.best_params_.get('max_leaf_nodes'))

#print(gsearch.best_params_.get('max_depth'))



'''final_model = DecisionTreeClassifier(max_leaf_nodes=gsearch.best_params_.get('max_leaf_nodes'),

                                     max_depth=gsearch.best_params_.get('max_depth'),

                                     random_state=1 )

final_model.fit(X, y)'''

#######################################################

final_model = RandomForestClassifier(

                         max_depth = gsearch.best_params_.get('max_depth'),

                           max_leaf_nodes = gsearch.best_params_.get('max_leaf_nodes'),

        n_estimators = gsearch.best_params_.get('n_estimators'),random_state=1, n_jobs=4)

final_model.fit(X, y)

#X.isna().sum()
data_test.Age.fillna(X.Age.mean(), inplace=True)

data_test.Fare.fillna(X.Fare.mean(), inplace=True)



data_test.isna().sum()
preds = final_model.predict(data_test)

print(preds.shape)

print(data_test.shape)
test_out = pd.DataFrame({

    'PassengerId': data_test.index, 

    'Survived': preds

})

test_out.to_csv('submission.csv', index=False)

print('Done')