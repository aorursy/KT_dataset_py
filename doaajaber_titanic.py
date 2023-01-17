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
t_data= pd.read_csv('/kaggle/input/titanic/train.csv',index_col='PassengerId')

t_data.head()
t_data
t_data.drop(columns=['Name','Ticket','Fare','Cabin'],inplace=True)
t_data.columns
for col in range(len( t_data.columns)):

    print (t_data[t_data.columns[col]].value_counts())
t_data.isna().sum()
t_data.Age.mean()
np.std(t_data.Age)
t_data.Age.value_counts().mode()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
t_data
target_col="Survived"

y = t_data[target_col]

X = t_data[['Pclass','Sex','Age','SibSp','Parch','Embarked']]



X = pd.get_dummies(X)

#X_test = pd.get_dummies(test_data[features])



train_X, val_X, train_y, val_y = train_test_split(X, y)

val_X

cols_with_missing = [col for col in train_X.columns  if train_X[col].isnull().any()]

red_X_train=train_X.drop(columns=cols_with_missing)

red_X_val=val_X.drop(columns=cols_with_missing)
def get_accuracy (n_estimators,max_depth,train_X, val_X, train_y, val_y):

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth ,random_state=1)

    model.fit(train_X,train_y)

    preds = model.predict(val_X)

    lr_accuracy = accuracy_score(val_y,preds)

    return lr_accuracy
accuracy=get_accuracy(200,10,red_X_train,red_X_val,train_y,val_y)

print("Validation accurcy for Random Forest Model: {}".format(accuracy))
from sklearn.impute import SimpleImputer

# Imputation

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X))

imputed_X_valid = pd.DataFrame(my_imputer.transform(val_X))



# Imputation removed column names; put them back

imputed_X_train.columns =train_X.columns

imputed_X_valid.columns = val_X.columns



accuracy=get_accuracy(1000,10,imputed_X_train,imputed_X_valid,train_y,val_y)

print("Validation accurcy for Random Forest Model: {}".format(accuracy))
max_accur=.5

max_dep=0

best_tree_size=0

for maxDepth in range(1,11):  

    for i in range(10,101,10):

        accuracy=get_accuracy(i,maxDepth,imputed_X_train,imputed_X_valid,train_y,val_y)

        if accuracy>max_accur:

            max_accur=accuracy

            max_dep=maxDepth

            best_tree_size=i

print("max accuracy = {}     max depth={}      best tree size={}".format(max_accur,max_dep,best_tree_size))
max_accur=.5

max_dep=0

best_tree_size=0

for maxDepth in range(1,11):  

    for i in range(10,101,10):

        accuracy=get_accuracy(i,maxDepth,red_X_train,red_X_val,train_y,val_y)

        if accuracy>max_accur:

            max_accur=accuracy

            max_dep=maxDepth

            best_tree_size=i

print("max accuracy = {}     max depth={}      best tree size={}".format(max_accur,max_dep,best_tree_size))
pd.get_dummies(df, prefix=['col1', 'col2'])
accuracy=get_accuracy(60,4,red_X_train,red_X_val,train_y,val_y)

accuracy
test_data= pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.info()

test_data
model = RandomForestClassifier(n_estimators=60, max_depth=10 ,random_state=1)

model.fit(imputed_X_train,train_y)

preds = model.predict(imputed_X_valid)

model_accuracy = accuracy_score(val_y,preds)

print("Accarany = {}:".format(model_accuracy))
test=test_data[['Pclass','Sex','Age','SibSp','Parch','Embarked']]

final_X_test = pd.get_dummies(test)



X_test.info()
final_X_test = pd.DataFrame(my_imputer.transform(final_X_test))


predictions = model.predict(final_X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")



'''output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")'''