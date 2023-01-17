import pandas as pd
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

combine = train.append(test)
X_train = train.drop(['Name','Ticket','Cabin','Survived','Embarked'], axis=1)

Y_train = train['Survived']



X_test = test.drop(['Name','Ticket','Cabin','Embarked'], axis=1)
X_train.shape,X_test.shape,Y_train.shape
test.head()
#X_train['Embarked']= X_train['Embarked'].astype('category')

#X_test['Embarked']= X_test['Embarked'].astype('category')



var = ['Pclass','SibSp','Parch','Sex']

from sklearn import preprocessing



le = preprocessing.LabelEncoder()



X_train = X_train[var].apply(le.fit_transform)

X_test = X_test[var].apply(le.fit_transform)





#X_train['Embarked']= X_train['Embarked'].astype('category')

#X_train['Embarked']=X_train['Embarked'].as

#le.transform(X_train['Embarked'])
from sklearn import ensemble

clf = ensemble.RandomForestClassifier(max_features='sqrt',n_estimators=50)



clf = clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_test)

submission = pd.DataFrame({'PassengerID':test.PassengerId,

                          'Survived':Y_pred})



submission.to_csv('submission.csv',index=False)