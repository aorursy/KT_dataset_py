import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier
df_train=pd.read_csv('../input/titanic/train.csv')

df_test=pd.read_csv('../input/titanic/test.csv')
passengerId=df_test['PassengerId']



y = df_train['Survived'].astype('int64')



df_train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], 1, inplace=True)

df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True)
sex = {'male':0, 'female':1, np.nan: np.nan}

embarked = {'Q':0, 'C':1, 'S':2, np.nan: np.nan}
def categorical_to_numerical(df):

    df.Sex = [sex[p] for p in df.Sex.values.tolist()]

    df.Embarked=[embarked[p] for p in df.Embarked.values.tolist()]
categorical_to_numerical(df_train)

categorical_to_numerical(df_test)
print(df_train.isnull().sum())

print(df_test.isnull().sum())
for column in df_train.columns:

    mean = df_train[column].mean()

    df_train[column].fillna(value=mean, inplace=True)



for column in df_test.columns:

    mean = df_test[column].mean()

    df_test[column].fillna(value=mean, inplace=True)
df_train = df_train.astype('int64')

df_test = df_test.astype('int64')
print(df_train.isnull().sum())

print(df_test.isnull().sum())
x_train, x_validate, y_train, y_validate = train_test_split(df_train, y, test_size=0.2, random_state=42)
# LogisticRegression

clf1=LogisticRegression(max_iter=1000)

clf1.fit(x_train, y_train)
acc=clf1.score(x_validate, y_validate)

print(acc)



predict1=clf1.predict(df_test)
# SVM

clf2=svm.SVC()

clf2.fit(x_train, y_train)
acc=clf2.score(x_validate, y_validate)

print(acc)



predict2=clf2.predict(df_test)
# KNeighborsClassifier

clf3=KNeighborsClassifier()

clf3.fit(x_train, y_train)
acc=clf3.score(x_validate, y_validate)

print(acc)



predict3=clf3.predict(df_test)
output1=pd.DataFrame({'PassengerId': passengerId, 'Survived': predict1})

output2=pd.DataFrame({'PassengerId': passengerId, 'Survived': predict2})

output3=pd.DataFrame({'PassengerId': passengerId, 'Survived': predict3})
output1.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")