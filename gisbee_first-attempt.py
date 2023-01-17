import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import pandas as pd

import numpy as np

import sklearn as sk

from sklearn import *



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/train.csv');df
classifier=df['Survived']

#one hot coding the embarked and sex columns

one_hot_embarked=pd.get_dummies(df['Embarked'],drop_first=True)

male=pd.get_dummies(df['Sex'],drop_first=True)

#joining dataframes together

df=pd.concat([df,one_hot_embarked,male],axis=1);df

#removing columns

df=df[list(set(df.columns)-set({'Name','Cabin','PassengerId','Survived','Sex','Embarked'}))];df

#removing text from ticket entries

df.Ticket=df.Ticket.str.split('\s+').str[-1];

df.Ticket[df.Ticket=='LINE']=0

#missing value imputation

df=df.fillna(df.mean())

df
test=pd.read_csv('../input/test.csv');test



#one hot coding the embarked and sex columns

one_hot_embarked=pd.get_dummies(test['Embarked'],drop_first=True)

male=pd.get_dummies(test['Sex'],drop_first=True)



#joining dataframes together

test=pd.concat([test,one_hot_embarked,male],axis=1)



#removing columns

test=test[list(set(test.columns)-set({'Name','Cabin','PassengerId','Survived','Sex','Embarked'}))]



#removing text from ticket entries

test.Ticket=test.Ticket.str.split('\s+').str[-1]

test.Ticket[test.Ticket=='LINE']=0

#missing value imputation

test=test.fillna(test.mean())



test
#running models



X_train, X_test, y_train, y_test =sk.model_selection.train_test_split(df,classifier)



clf=sk.ensemble.RandomForestClassifier(n_estimators=1000,max_depth=20)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)



print(sk.metrics.accuracy_score(y_test,y_pred))





y_pred=clf.predict(X_test)



print(sk.metrics.accuracy_score(y_test,y_pred))

clf=sk.naive_bayes.GaussianNB()

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)



print(sk.metrics.accuracy_score(y_test,y_pred))

clf=sk.ensemble.AdaBoostClassifier(n_estimators=1000)

clf.fit(df,classifier)

prediction = clf.predict(test)
ident=[]

for i in range(0,len(prediction)):

    ident.append(892+i)

submission=pd.DataFrame(prediction,index=ident,columns=['Survived'])

submission.index.names=['PassengerId']

submission.to_csv('submission.csv')