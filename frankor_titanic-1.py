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

import statistics
#Read data

train_set = pd.read_csv('../input/titanic/train.csv')

test_set = pd.read_csv('../input/titanic/test.csv')

train_set.head()
def preprocess(df):

    """Fares are categorised into 1 (any fare paid) and 0 (no fare, suggesting crew)

    Cabins, where applicable, are categorised by block (first letter)

    Missing ages are filled with the group median

    """

    transform = df

    ages = pd.notnull(df.Age)

    avg_age = sum(ages)/len(ages)

    values = {'Age': avg_age,}

    transform.fillna(value=values,inplace=True)

    for i in range(len(df.Cabin)):

        cabincode = str(df.Cabin[i])

        transform.Cabin[i] = cabincode[0]

        if df.Fare[i] > 0:

            transform.Fare[i]=1

        else:

            transform.Fare[i]=0

    return transform

train_transform = preprocess(train_set)

test_transform = preprocess(test_set)
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
# Remove unwanted columns

y = train_set['Survived']

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin']

X = pd.get_dummies(train_transform[features])

X_test = pd.get_dummies(test_transform[features])

X.head()

X_test.head()

# We observe the training data contains entries with cabin "T", while the test data does not; drop the column

X.drop(['Cabin_T'],axis=1,inplace=True)
missing_val_count_by_column = (X.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
#Fit the RF model

X_train,X_val,y_train,y_val = train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=0)

model = XGBRegressor(n_estimators=400,learning_rate=0.04)

model.fit(X_train,y_train,early_stopping_rounds=5,eval_set=[(X_val,y_val)])
# Apply model and round predictions from decimals to 1/0

predictions = model.predict(X_test)

predict_votes = [int(round(prediction)) for prediction in predictions]
output = pd.DataFrame({'PassengerId': test_set.PassengerId, 'Survived': predict_votes})

output.to_csv('submission_titanic_1.csv', index=False)

print('check123')