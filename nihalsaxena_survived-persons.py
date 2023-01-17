import pandas as pd
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


ids = test_df['PassengerId']
print(train_df.columns.values)
train_df.head()
train_df.tail()
train_df.info()
print('_'*40)
test_df.info()
train_df.describe()
train_df=train_df.drop(columns=['Name','Cabin','Ticket','PassengerId'])

test_df=test_df.drop(columns=['Name','Cabin','Ticket','PassengerId'])

train_df.shape
survived = train_df['Survived']

features = train_df.drop('Survived',axis=1)
from sklearn.preprocessing import Imputer
my_imputer = Imputer('NaN','median')
features['Age'] = my_imputer.fit_transform(features[['Age']])
test_df['Age'] = my_imputer.fit_transform(test_df[['Age']])

features['Fare'] = my_imputer.fit_transform(features[['Fare']])
test_df['Fare'] = my_imputer.fit_transform(test_df[['Fare']])
features=pd.get_dummies(features)

test_df=pd.get_dummies(test_df)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(survived)

survived = le.transform(survived)
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    survived,random_state=0)
def prediction(Model,X_train,y_train,X_test,y_test) :
    
    clf=Model()
    
    clf.fit(X_train,y_train)
    
    print(clf.score(X_test,y_test))
    
    return clf

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.neural_network import MLPClassifier

clf_A = prediction(RandomForestClassifier,X_train,y_train,X_test,y_test)

clf_B = prediction(MLPClassifier,X_train,y_train,X_test,y_test)

clf_C = prediction(AdaBoostClassifier,X_train,y_train,X_test,y_test)
predictions = clf_C.predict(test_df)


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions.csv', index = False)

