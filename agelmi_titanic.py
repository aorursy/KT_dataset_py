import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn import svm

from sklearn.preprocessing import LabelEncoder



#load the data

data = pd.read_csv('../input/titanic/train.csv').fillna('')

data
#preprocessing

#We encode non-numerical data into integers using LabelEncoder

enc = LabelEncoder()

data['Name'] = enc.fit_transform(data['Name'].values)

data['Sex'] = enc.fit_transform(data['Sex'].values)

data['Ticket'] = enc.fit_transform(data['Ticket'].values)

# Deal with NaN values

ages = data['Age'].values

s=0

c=0

#there must be a smarter way to do this :\

for a in ages:

    try:

        a = float(a)

        s+=a

        c+=1

    except:

        continue

avg = s/c

data['Age'] = [age if age != '' else avg for age in ages]

data['Age'] = data.Age.astype(float)

data['Cabin'] = enc.fit_transform(['Empty' if x != x else x for x in data['Cabin'].values])

data['Embarked'] = enc.fit_transform(['Uknown' if x != x else x for x in data['Embarked'].values])



X = data.drop('Survived',axis=1).values

y = data['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)



#fix predictive model, in this case I chose linear SVM

model = svm.SVC(kernel='linear')

model.fit(X_train, y_train)

accuracy_score(y_test,model.predict(X_test))
#preprocessing

#in this case we just drop all rows with any NaN value

data = pd.read_csv('../input/titanic/train.csv').dropna()

data['Name'] = enc.fit_transform(data['Name'].values)

data['Sex'] = enc.fit_transform(data['Sex'].values)

data['Ticket'] = enc.fit_transform(data['Ticket'].values)

data['Cabin'] = enc.fit_transform(['Empty' if x != x else x for x in data['Cabin'].values])

data['Embarked'] = enc.fit_transform(['Uknown' if x != x else x for x in data['Embarked'].values])



X = data.drop('Survived',axis=1).values

y = data['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)



#fix predictive model, in this case I chose linear SVM

#fix predictive model, in this case I chose linear SVM

model = svm.SVC(kernel='linear')

model.fit(X_train, y_train)

accuracy_score(y_test,model.predict(X_test))

#Notice the accuracy is not much different than before
#Feature selection with Recursive Feature Elimination

from sklearn.feature_selection import RFE

selector = RFE(model, 5) # select the 5 best features

selector = selector.fit(X_train, y_train)

headers_to_keep = [h for (h,s) in zip(list(data.drop('Survived',axis=1)),selector.support_) if s] # headers of the selected features

headers_to_keep
data = pd.read_csv('../input/titanic/train.csv').dropna()

y = data['Survived'].values

data = data.drop([h for h in list(data) if h not in headers_to_keep], axis=1)

data['Sex'] = enc.fit_transform(data['Sex'].values)

X = data.values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)



model.fit(X_train, y_train)

accuracy_score(y_test,model.predict(X_test))