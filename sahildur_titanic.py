print("in progress")
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

test_label_df = pd.read_csv("../input/gendermodel.csv")

train_df.head()
train_df[train_df['Survived']==1]['Age'].hist(bins=20)
train_df['Age'].hist(bins=20)
train_labels=[]

train_gender=[]

train_child=[]

for index, row in train_df.iterrows():    

    train_labels.append(row['Survived'])

    if(row['Sex']=='male'):

        train_gender.append('1')

    else:

        train_gender.append('2')

    if(row['Age']<=15):

        train_child.append('1')

    else:

        train_child.append('0')
test_labels=[]

test_gender=[]

test_child=[]

for index, row in test_df.iterrows():

    #test_labels.append(row['Survived'])

    if(row['Sex']=='male'):

        test_gender.append('1')

    else:

        test_gender.append('2')

    if(row['Age']<=15):

        test_child.append('1')

    else:

        test_child.append('0')

        

for index, row in test_label_df.iterrows():

    test_labels.append(row['Survived'])
print(train_gender[0:5])

print(train_labels[0:5])

print(train_child[0:5])



print("len")

print(len(train_gender))

a = np.array((train_gender))

b = np.array((train_labels))

d = np.array((train_child))

c = np.array((train_df['Pclass']))

train_inputs=np.stack((a, d, c), axis=-1)
p = np.array((test_gender))

q = np.array((test_labels))

r = np.array((test_child))

s = np.array((test_df['Pclass']))

test_inputs=np.stack((p, r, s), axis=-1)
from sklearn.ensemble import RandomForestClassifier

X = train_inputs

Y = train_labels

clf = RandomForestClassifier(n_estimators=10)

clf = clf.fit(X, Y)
train_labels_np=b

train_pred = clf.predict(train_inputs)

train_pred.reshape(-1, 1)[0:5]

#clf.score(train_labels_np.reshape(-1, 1), train_pred.reshape(-1, 1))

from sklearn.metrics import accuracy_score

print("Training accuracy is " + str(accuracy_score(train_labels_np, train_pred)*100) + "%")
test_labels_np=q

test_pred = clf.predict(test_inputs)

test_pred.reshape(-1, 1)[0:5]

#clf.score(train_labels_np.reshape(-1, 1), train_pred.reshape(-1, 1))

from sklearn.metrics import accuracy_score



print("Testing accuracy is " + str(accuracy_score(test_labels_np, test_pred)*100) + "%")
print(train_labels_np[0:10])

print(train_pred[0:10])

print(test_labels_np[0:10])

print(test_pred[0:10])
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": test_pred

    })

submission.to_csv('titanic.csv', index=False)