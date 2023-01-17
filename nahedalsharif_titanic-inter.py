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
filepath='../input/titanic/train.csv'

home_data=pd.read_csv(filepath, index_col='PassengerId')

home_data


home_data.drop('Name', inplace= True,axis=1)

home_data.drop('Ticket', inplace= True,axis=1)

home_data.drop('Cabin', inplace= True,axis=1)

home_data.Sex.replace('male',0, inplace=True)

home_data.Sex.replace('female',1, inplace=True)

home_data.Embarked.replace('S',0, inplace=True)

home_data.Embarked.replace('C',1, inplace=True)

home_data.Embarked.replace('Q',2, inplace=True)

home_data
X=home_data.drop('Survived', axis=1)

y=home_data.Survived



from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y=train_test_split(X,y)



train_X.shape + val_X.shape

X
# Drop Nan columns

cols_with_missing = [col for col in train_X.columns

                     if train_X[col].isnull().any()]

cols_with_missing
reduced_X = train_X.drop(cols_with_missing, axis=1)

reduced_val= val_X.drop(cols_with_missing, axis=1)

reduced_X.shape+reduced_val.shape
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score 



print ('Accuracy of * Drop columns & DecisionTree model *')

DTpara=list(range(2,15,1))

for para in DTpara:

    DTmodel= DecisionTreeClassifier(max_leaf_nodes=para)

    DTmodel.fit(reduced_X,train_y)

    preds=DTmodel.predict(reduced_val)

    acc= accuracy_score(y_true=val_y, y_pred=preds)

    print(para,'   ', acc)

from sklearn.ensemble import RandomForestClassifier





print ('Accuracy of * Drop columns & RandomForest model *')



RFpara=list(range(50,200,25))



for para in RFpara:                  

    RFmodel=RandomForestClassifier(n_estimators=para) 

    RFmodel.fit(reduced_X,train_y)

    preds=RFmodel.predict(reduced_val)

    RFacc= accuracy_score(y_true=val_y, y_pred=preds)

    print(para,'   ', acc)
# Imputation

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))

imputed_val_X = pd.DataFrame(my_imputer.transform(val_X))



# Imputation removed column names; put them back



imputed_train_X.columns = train_X.columns

imputed_val_X.columns = val_X.columns



#print("MAE from Approach 2 (Imputation):")

#print(score_dataset(imputed_train_X, imputed_val_X, train_y, val_y))
imputed_train_X.isnull().sum()

imputed_val_X.isnull().sum()
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score 



print ('Accuracy of * Imputation & DecisionTree model *')

DTpara=list(range(2,15,1))

for para in DTpara:

    DTmodel= DecisionTreeClassifier(max_leaf_nodes=para)

    DTmodel.fit(imputed_train_X,train_y)

    preds=DTmodel.predict(imputed_val_X)

    acc= accuracy_score(y_true=val_y, y_pred=preds)

    print(para,'   ', acc)
from sklearn.ensemble import RandomForestClassifier





print ('Accuracy of * Imputation & RandomForest model *')



RFpara=list(range(50,200,25))



for para in RFpara:                  

    RFmodel=RandomForestClassifier(n_estimators=para) 

    RFmodel.fit(imputed_train_X,train_y)

    preds=RFmodel.predict(imputed_val_X)

    RFacc= accuracy_score(y_true=val_y, y_pred=preds)

    print(para,'   ', acc)
#Imputaion of fullData



#from sklearn.impute import SimpleImputer

my_imputer1 = SimpleImputer()

imputed_full_X = pd.DataFrame(my_imputer1.fit_transform(X))

# Imputation removed column names; put them back

imputed_full_X.columns = X.columns



imputed_full_X 
#Train best model with fullData



finalmodel= RandomForestClassifier(n_estimators=175)

finalmodel.fit(imputed_full_X,y)
testfilepath='../input/titanic/test.csv'

test_data=pd.read_csv(testfilepath, index_col='PassengerId')





test_data.drop(columns=['Name','Cabin','Ticket'], inplace=True)

test_data.Sex.replace('male',0, inplace=True)

test_data.replace('female',1, inplace=True)

test_data.Embarked.replace('S',0, inplace=True)

test_data.Embarked.replace('C',1, inplace=True)

test_data.Embarked.replace('Q',2, inplace=True)

test_data
#Imputation Test Data

from sklearn.impute import SimpleImputer

my_imputer2 = SimpleImputer()

my_imputer2.fit(train_X)

imputed_test = pd.DataFrame(my_imputer2.transform(test_data))

imputed_test.columns=test_data.columns



imputed_test
finalpreds=finalmodel.predict(imputed_test)

finalpreds
output = pd.DataFrame({'PassengerId': test_data.index,

                       'Survived': finalpreds})

output
output.to_csv('submission.csv', index=False)

print('done')