# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')

import numpy as np

from sklearn.cluster import KMeans

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier
df.head()
df.drop(['Name'], axis=1, inplace = True)

df.drop(['PassengerId'], axis=1, inplace= True)

test.drop(['Name'], axis=1, inplace = True)

df.fillna(0, inplace = True)

test.fillna(0, inplace = True)

# Bins for ages

# Labels to categorize the ages.

bins = [0,1, 5, 10, 25, 50, 100]

labels = [1,2,3,4,5,6]
df['Age'] = pd.cut(df['Age'], bins = bins, labels = labels)

test['Age'] = pd.cut(test['Age'], bins = bins, labels = labels)

df = pd.get_dummies(df, columns = ['Sex'],drop_first = True )

test = pd.get_dummies(test, columns = ['Sex'],drop_first = True )
# Creating labels and bins for fare column

bins = [0,10,20,30,50, 100, 200 , 250, 300, 350, 400, 450, 500, 550]

labels = [1,2,3,4,5,6,7,8,9,10,11,12,13]

df['Fare'] = pd.cut(df['Fare'], bins = bins, labels = labels)

test['Fare'] = pd.cut(test['Fare'], bins = bins, labels = labels)

df['Fare'] = df['Fare'].astype('int32')

test['Fare'] =  test['Fare'].astype('int32')
fare_scale = preprocessing.MinMaxScaler()



df_fares = df['Fare'].values

scaled_fares = df_fares.reshape(-1,1)

scaled_fares = fare_scale.fit_transform(scaled_fares)

scaled_fares = scaled_fares.flatten()

df['Fare'] = pd.Series(scaled_fares)



test_fares = test['Fare'].values

scaled_fares = test_fares.reshape(-1,1)

scaled_fares = fare_scale.transform(scaled_fares)

scaled_fares = scaled_fares.flatten()

test['Fare'] = pd.Series(scaled_fares)
df.head()
label_enc = preprocessing.LabelEncoder()

df['Cabin'] = df['Cabin'].astype('str')

test['Cabin'] = test['Cabin'].astype('str')



enc_list = []

for i in df['Cabin'].values:

    enc_list.append(i)

    

for i in test['Cabin'].values:

    enc_list.append(i)

    

label_enc.fit(enc_list)

    

df['Cabin'] = label_enc.transform(df['Cabin'])



test['Cabin'] = label_enc.transform(test['Cabin'])
label_enc = preprocessing.LabelEncoder()

df['Embarked'] = df['Embarked'].astype('str')

df['Embarked'] = label_enc.fit_transform(df['Embarked'])



test['Embarked'] = test['Embarked'].astype('str')

test['Embarked'] = label_enc.transform(test['Embarked'])
label_enc = preprocessing.LabelEncoder()

df['Ticket'] = df['Ticket'].astype('str')

test['Ticket'] = test['Ticket'].astype('str')



enc_list = []

for i in df['Ticket'].values:

    enc_list.append(i)

    

for i in test['Ticket'].values:

    enc_list.append(i)

    

label_enc.fit(enc_list)

    

df['Ticket'] = label_enc.transform(df['Ticket'])



test['Ticket'] = label_enc.transform(test['Ticket'])
df = df.astype('float')

df.fillna(0, inplace = True)



test = test.astype('float')

test.fillna(0, inplace = True)
df.describe()
X = np.array(df.drop(['Survived'], axis = 1))

X = preprocessing.scale(X)

# If we avoid this preprocessing then it will hurt a lot in SVM

Y = np.array(df['Survived'])
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,stratify = Y, test_size = 0.1, random_state = 31)
model = svm.SVC(kernel = 'poly',degree=3, random_state = 31, gamma = "auto", C = 1)

model.fit(X_train, Y_train)

predictions = model.predict(X_train)

accuracy = accuracy_score(predictions,Y_train)

print("Training accuracy = %0.2f" %(accuracy * 100))
predictions = model.predict(X_test)

accuracy = accuracy_score(predictions,Y_test)

print("Testing accuracy = %0.2f" %(accuracy * 100))
clf = DecisionTreeClassifier(max_depth=4, random_state = 31)

clf.fit(X_train, Y_train)

predictions = model.predict(X_train)

accuracy_clf = accuracy_score(predictions, Y_train)

print("Training accuracy = %0.2f" %(accuracy_clf * 100))
predictions = clf.predict(X_test)

accuracy_clf = accuracy_score(predictions, Y_test)

print("Testing accuracy = %0.2f" %(accuracy_clf * 100))
test.head()
Pid = test['PassengerId']

X_final_test = test.drop(['PassengerId'], axis = 1)

X_final_test = preprocessing.scale(X_final_test)

Y_pred = clf.predict(X_final_test)
test['Survived'] = Y_pred
test.head()
test = test.astype('int32')
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": test["Survived"]

    })
submission.to_csv('submission.csv', index=False)