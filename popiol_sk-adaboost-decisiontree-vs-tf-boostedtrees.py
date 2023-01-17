import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import math

import tensorflow as tf

from tensorflow import keras

import itertools
train = pd.read_csv('../input/train.csv')

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
all = pd.concat([train,test], sort=False, ignore_index=True)

all.head()
all_cabin1 = all['Cabin'].apply(lambda x: str(x)[:1] if x == x else '')

all_cabin1 = pd.get_dummies(all_cabin1);

all_cabin1.head()
all_cabin2 = all['Cabin'].apply(lambda x: str(x).split(' ')[0][1:] if x == x else x)

all_cabin2 = all_cabin2.apply(lambda x: int(x) if str(x).isdigit() else math.nan)

all_cabin2.head()
all_cabin3 = all['Cabin'].str.len()

all_cabin3.head()
all_ticket1 = all['Ticket'].apply(lambda x: (str(x).split(' ')[0] if len(str(x).split(' ')) > 1 else '') if x == x else '')

all_ticket1 = pd.get_dummies(all_ticket1.str.replace('.','').str.replace('/','').str.upper())

all_ticket1.head()
all_ticket2 = all['Ticket'].apply(lambda x: str(x).split(' ')[len(str(x).split(' '))-1] if x == x else '')

all_ticket2 = all_ticket2.apply(lambda x: int(x) if str(x).isdigit() else math.nan)

all_ticket2.head()
all_name1 = all['Name'].apply(lambda x: 1 if '(' in x else 0)

all_name1.head()
all_name2 = all['Name'].str.len()

all_name2.head()
all_agefare = all.apply(lambda row: row['Fare'] / row['Age'], axis=1)

all_agefare.head()
all_sex = pd.Series(LabelEncoder().fit_transform(all["Sex"]))

all_sex.head()
all_embarked = pd.get_dummies(all['Embarked'])

all_embarked.head()
X = all.drop(columns=["PassengerId","Name","Sex","Ticket","Cabin","Embarked"])

X["Sex"] = all_sex

for c in all_embarked:

    X["Embarked_"+c] = all_embarked[c]

for c in all_ticket1:

    X["Ticket1_"+c] = all_ticket1[c]

for c in all_cabin1:

    X["Cabin1_"+c] = all_cabin1[c]

X["Cabin2"] = all_cabin2

X["Cabin3"] = all_cabin3

X["Ticket2"] = all_ticket2

X["Name1"] = all_name1

X["Name2"] = all_name2

X['Fare_Age'] = all_agefare



X.head()
for c in list(X.drop(columns="Survived")):

    X[c] = X[c].apply(lambda x: X[c].median() if x != x else x)



X.head()
for c in list(X.drop(columns="Survived")):

    X[c] = pd.cut(X[c],100)



X.head()
for c in list(X.drop(columns="Survived")):

    X[c] = X[c].apply(lambda x: x.right)

    X[c] = pd.Series(LabelEncoder().fit_transform(X[c]))



X.head()
trainX = X.loc[X["Survived"] == X["Survived"]]

trainY = trainX.pop("Survived")

trainX.head()
trainY.head()
testX = X.loc[X["Survived"] != X["Survived"]].drop(columns="Survived")

testX.head()
sampleX = []

sampleY = []

sampleXtest = []

sampleYtest = []

all = pd.concat([trainX,trainY],axis=1)



for i in range(0, 9):

    sampleX.append(all.sample(n=math.ceil(len(trainX)/10), random_state=i))

    sampleXtest.append(all.iloc[all.index.difference(sampleX[i].index)])

    sampleX[i] = pd.DataFrame(sampleX[i].values, columns=sampleX[i].columns)

    sampleXtest[i] = pd.DataFrame(sampleXtest[i].values, columns=sampleXtest[i].columns)

    sampleY.append(sampleX[i].pop('Survived'))

    sampleYtest.append(sampleXtest[i].pop('Survived'))

    

sampleY[0].head()
def evaluate(Y_pred, testY):

    err = sum(abs(y - Y_pred[i]) for i, y in testY.iteritems())

    return 1-err/testY.count()
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), learning_rate=.9, algorithm='SAMME')
score = []

for i in range(0, 9):

    model.fit(sampleX[i],sampleY[i])

    Y_pred = pd.Series(model.predict(sampleXtest[i])).astype(int)

    score.append(evaluate(Y_pred=Y_pred,testY=sampleYtest[i]))

    

(np.mean(score) + np.min(score))/2
def train_input_fn(trainX, trainY, shuffle=0):

    dataset = tf.data.Dataset.from_tensor_slices((dict(trainX), trainY.values))

    for _ in itertools.repeat(None, shuffle):

        dataset = dataset.shuffle(100).repeat()

    dataset = dataset.batch(10)

    return dataset
def test_input_fn(testX):

    dataset = tf.data.Dataset.from_tensor_slices(dict(testX))

    dataset = dataset.batch(10)

    return dataset
feature_columns = [tf.feature_column.numeric_column(name) for name in trainX.columns]
score = []

for i in range(0, 9):

    model = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=20)

    model.train(input_fn=(lambda:train_input_fn(trainX=sampleX[i],trainY=sampleY[i],shuffle=4)),max_steps=1000)

    pred = model.predict(input_fn=lambda:test_input_fn(testX=sampleXtest[i]))

    Y_pred = pd.Series([int(next(pred)['classes'][0]) for _ in itertools.repeat(None, len(sampleXtest[i]))])

    score.append(evaluate(Y_pred=Y_pred,testY=sampleYtest[i]))



(np.mean(score) + np.min(score))/2
# model = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=20)

# model.train(input_fn=(lambda:train_input_fn(trainX, trainY)),max_steps=1000)

# pred = model.predict(input_fn=lambda:test_input_fn(testX))

# Y_pred = pd.Series([int(pred.next()['classes'][0]) for _ in itertools.repeat(None, len(testX))])
# model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), learning_rate=.9, algorithm='SAMME')

# model.fit(trainX,trainY)

# Y_pred = pd.Series(model.predict(trainX)).astype(int)
# Y_pred.head()
# submission = pd.DataFrame({

#     "PassengerId": test["PassengerId"],

#     "Survived": Y_pred

# })



# submission.head()
# submission.to_csv('submission.csv', index=False)