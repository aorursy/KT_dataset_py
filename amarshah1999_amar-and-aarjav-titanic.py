# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.impute import SimpleImputer

from sklearn import preprocessing

from sklearn.model_selection import train_test_split, cross_val_score



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.info()

print("")

df_test.info()
df_train.head()
#2 null values in embarked can be dropped

df_train = df_train.dropna(subset = ['Embarked']).reset_index(drop = True)

columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']

df_train = df_train.drop(columns_to_drop, axis = 1)

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
#impute

imputer = SimpleImputer(strategy = 'mean', missing_values = np.nan)



df_train['Age'] = imputer.fit_transform(df_train[['Age']])

df_test['Age'] = imputer.transform(df_test[['Age']])

df_train['Fare'] = imputer.fit_transform(df_train[['Fare']])

df_test['Fare'] = imputer.transform(df_test[['Fare']])
#encode categoricals

columns_to_encode =['Sex', 'Embarked']

ohe = preprocessing.OneHotEncoder(sparse = False)

encoded_cols = pd.DataFrame(ohe.fit_transform(df_train[columns_to_encode]))

df_train = df_train.drop(columns_to_encode, axis= 1)

df_train = pd.concat([df_train, encoded_cols], axis = 1)



test_encoded_cols = pd.DataFrame(ohe.transform(df_test[columns_to_encode]))

df_test = pd.concat([df_test.drop(columns_to_encode, axis = 1), test_encoded_cols], axis = 1)

df_train.rename(columns = {0:'gender_0', 1:'gender_1', 2:'port_1', 3:'port_2', 4:'port_3'}, inplace=True)

df_test.rename(columns = {0:'gender_0', 1:'gender_1', 2:'port_1', 3:'port_2', 4:'port_3'}, inplace=True)
#feature scaling

columns_to_scale = ['Age', 'Fare', 'SibSp','Parch']

df_train[columns_to_scale] = preprocessing.scale(df_train[columns_to_scale])

df_test[columns_to_scale] = preprocessing.scale(df_test[columns_to_scale])
x = df_train.drop(['Survived'], axis = 1)

y = df_train.Survived



rf = RandomForestClassifier()

lr = LogisticRegression()

svc = SVC()

xgb = XGBClassifier()

lr = LogisticRegression()



def modeltest(model, x_train, y_train):

    cv = cross_val_score(model, x_train, y_train, cv = 5)

    print(cv)

    print(cv.mean())



modeltest(rf,x,y)

modeltest(lr,x,y)

modeltest(xgb,x,y)

modeltest(svc,x,y)



#submission code

#test_submission = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Survived':svc.predict(df_test.drop(['PassengerId'], axis = 1))})

#test_submission.to_csv('submission.csv', index = False)
