import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
train_data = pd.read_csv('../input/titanic/train.csv',index_col='PassengerId')

test_data = pd.read_csv('../input/titanic/test.csv',index_col='PassengerId')



train_data.Age.fillna(value = np.mean(train_data.Age),inplace=True)

test_data.Age.fillna(value = np.mean(train_data.Age),inplace=True)



train_data.Fare.fillna(value = np.mean(train_data.Fare),inplace=True)

test_data.Fare.fillna(value = np.mean(train_data.Fare),inplace=True)



train_data.fillna(value='',inplace=True)

test_data.fillna(value='',inplace=True)
X_train = train_data.drop(columns=['Name','Survived','Ticket'])

y_train = train_data['Survived']

X_test = test_data.drop(columns=['Name','Ticket'])
def Cabin_numbers(cabin):

    if cabin == '':

        return 1

    else :

        return len(cabin.split())

    

def Cabin_type(cabin):

    if cabin == '':

        return 0

    else:

        return ord(cabin[0])-ord('A')
X_train = X_train.assign(Cabin_nums = X_train['Cabin'].apply(Cabin_numbers),Cabin_type = X_train['Cabin'].apply(Cabin_type))

X_test = X_test.assign(Cabin_nums = X_test['Cabin'].apply(Cabin_numbers),Cabin_type = X_test['Cabin'].apply(Cabin_type))



X_train.drop(columns = ['Cabin'],inplace=True)

X_test.drop(columns = ['Cabin'],inplace=True)



cat_features = ['Sex', 'Embarked']

encoder = LabelEncoder()



for col in cat_features:

    encoded = encoder.fit_transform(X_train[col])

    X_train[col]=encoded

    encoded = encoder.transform(X_test[col])

    X_test[col]=encoded
clf_gini = RandomForestClassifier()

clf_gini.fit(X_train, y_train)

y_predict = clf_gini.predict(X_test)
prediction = pd.DataFrame(data = {'PassengerId' : test_data.index, 'Survived' : y_predict })
prediction.to_csv('gender_submission.csv',index=False)