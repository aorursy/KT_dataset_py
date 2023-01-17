import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')



def process_cabin(x):

    x = str(x)

    if 'a' in x:

        return 11.0

    elif 'b' in x:

        return 12.0

    elif 'c' in x:

        return 13.0

    elif 'd' in x:

        return 14.0

    elif 'e' in x:

        return 15.0

    else: 

        return len(str(x))



def transform(db):

    db['Age'].fillna(34.0, inplace=True) 

    db['Cabin'].fillna('', inplace=True)     

    db['Age_bin'] = pd.cut(db['Age'], [0., 1.0, 5.0, 18.0, 33.0, 45.0, 65.0, 120.0], 

                       labels=False)    

    db = db[['Sex', 'Ticket', 'Cabin', 'Pclass', 'Age_bin', 'SibSp',

       'Parch', 'Fare', 'Embarked','Age']]

    db['Sex'] = db['Sex'].replace({'male':5,'female':-5})

    db['Ticket'] = db['Ticket'].apply(len)

    db['Cabin'] = db['Cabin'].apply(process_cabin)

    db['Embarked'] = db['Embarked'].replace({'S':7,'Q':10,'C':-5})

    return db 

    



from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import accuracy_score

import statistics

test = pd.read_csv('/kaggle/input/titanic/test.csv')



X, y = transform(df), df['Survived']



N, results, predictions = 30, [], []



for i in range(N):

   

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model = xgb.XGBClassifier(n_estimators=1000,max_depth=3,reg_lambda=2.0)



    model.fit(X_train, y_train)

    y_prediction = model.predict(X_val)



    results.append(accuracy_score(y_val,y_prediction))

    predictions.append(model.predict(transform(test)))

    

print(sum(results)/N, statistics.stdev(results))



rating = []

for i in range(len(test)):

     rating.append(sum([pred[i] for pred in predictions])) 



rating = [1 if rate>16 else 0 for rate in rating]



y_prediction = rating        

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

submission['Survived'] = y_prediction

submission.to_csv('my_results.csv',index=False)