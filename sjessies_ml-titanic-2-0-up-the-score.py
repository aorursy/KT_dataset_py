# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load of test and training data

train_raw_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_raw_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# Exploring and understand the data set

train_raw_data.head()
# Scan training data, null data count

print("Train data")

train_raw_data.isnull().sum()
# Mapping unique values from Embarked

lookup_Embarked_values = dict(zip(train_raw_data.Embarked.unique(), train_raw_data.Embarked.unique()))   

lookup_Embarked_values
# Clean data process 

def clean_data(df):

    # Fill de Nan Valuen in Fare an Age 

    df["Fare"]=df["Fare"].fillna(df["Fare"].dropna().median())

    # Fill the Nan values in Age Column

    df["Age"]=df["Age"].fillna(df["Age"].dropna().median())

    #Clear NaN values from embarked:

    df.dropna(subset=['Embarked'],inplace=True)

    #Drop Columns no used to do the analsis

    df.drop(['Cabin','Name','PassengerId','Ticket'],axis=1,inplace=True)

    #Change Male and female por numeric values

    df.Sex = [1 if each == "male" else 0 for each in df.Sex]

    #Change Embarked por numeric values

    df.loc[df["Embarked"]=="S","Embarked"]=0

    df.loc[df["Embarked"]=="C","Embarked"]=1

    df.loc[df["Embarked"]=="Q","Embarked"]=2

    

    return df



clean_train_data = clean_data(train_raw_data)
clean_train_data.isnull().sum()
clean_train_data.head()
#Split de train clean data in X(data) y Y(Label):

X = clean_train_data.drop('Survived', axis=1)

y = clean_train_data.get('Survived')
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



clf = LogisticRegression().fit(X_train, y_train)

print('Breast cancer dataset')

print('Accuracy of Logistic regression classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
test_raw_data.head()
#Exploración y limpieza de los datos de prueba:

print("Test data")

test_raw_data.isnull().sum()
# Clean data process 

def clean_data(df):

    # Fill de Nan Valuen in Fare an Age 

    df["Fare"]=df["Fare"].fillna(df["Fare"].dropna().median())

    # Fill the Nan values in Age Column

    df["Age"]=df["Age"].fillna(df["Age"].dropna().median())

    #Clear NaN values from embarked:

    df.dropna(subset=['Embarked'],inplace=True)

    #Drop Columns no used to do the analsis

    df.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)

    #Change Male and female por numeric values

    df.Sex = [1 if each == "male" else 0 for each in df.Sex]

    #Change Embarked por numeric values

    df.loc[df["Embarked"]=="S","Embarked"]=0

    df.loc[df["Embarked"]=="C","Embarked"]=1

    df.loc[df["Embarked"]=="Q","Embarked"]=2

    

    return df
clean_test_data = clean_data(test_raw_data)

clean_test_data.isnull().sum()
clean_test_data.head()
# Add the column ['predicted_survived'] to data set

clean_test_data['predicted_survived'] = clf.predict(clean_test_data.drop('PassengerId',axis=1))
clean_test_data.head()
submission = pd.DataFrame({

        "PassengerId": clean_test_data['PassengerId'],

        "Survived": clean_test_data['predicted_survived']

    })

submission.to_csv('submission.csv',index=False)