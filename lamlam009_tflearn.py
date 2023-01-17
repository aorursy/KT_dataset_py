# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tflearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

X = df.copy()
columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]

X = X[columns]



for i in columns:

    X = X[~(X[i].isnull())]

    

for i in range(1,4):

    X["Pclass_" + str(i)] = X["Pclass"] == i

del X["Pclass"]



for i in X.Embarked.unique():

    X[i] = X["Embarked"] == i

del X["Embarked"]



for i in X.Sex.unique():

    X[i] = X["Sex"] == i

del X["Sex"]



y = pd.DataFrame({"Survived":X["Survived"], "Not Survived":(1-X["Survived"])})

y.shape

del X["Survived"]
X = np.array(X, dtype = np.float32)

y = np.array(y, dtype = np.float32)
# Build neural network

net = tflearn.input_data(shape=[None, 12])

net = tflearn.fully_connected(net, 32)

net = tflearn.fully_connected(net, 32)

net = tflearn.fully_connected(net, 2, activation='softmax')

net = tflearn.regression(net)
# Define model

model = tflearn.DNN(net)

# Start training (apply gradient descent algorithm)

model.fit(X, y, n_epoch=200, batch_size=16, show_metric=True)
df = pd.read_csv("../input/test.csv")

X_test = df.copy()
columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X_test = X_test[columns]

    

for i in range(1,4):

    X_test["Pclass_" + str(i)] = X_test["Pclass"] == i

del X_test["Pclass"]



for i in X_test.Embarked.unique():

    X_test[i] = X_test["Embarked"] == i

del X_test["Embarked"]



for i in X_test.Sex.unique():

    X_test[i] = X_test["Sex"] == i

del X_test["Sex"]
X_test = np.array(X_test, dtype=np.float32)
from sklearn.preprocessing import Imputer

imputer = Imputer()

X_test = imputer.fit_transform(X_test)
pred = model.predict(X_test)
predict = np.zeros(len(pred))

for i in range(len(pred)):

    if (pred[i][1] >= 0.5):

        predict[i] = 1
y_test = pd.read_csv("../input/gender_submission.csv")

y_test = pd.DataFrame({"Survived":y_test["Survived"], "Not Survived":(1-y_test["Survived"])})

y_test = np.array(y_test, dtype = np.float32)
model.evaluate(X_test, y_test, batch_size=16)
test = pd.read_csv("../input/test.csv")

my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predict.astype(int)})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df_ans = pd.read_csv("../input/gender_submission.csv")
df_ans.head()
df_test.head()
del df_test["PassengerId"]

result = pd.concat([df_ans, df_test], axis=1, join_axes=[df_ans.index])
result
df_train
frames = [df_train, result]

result = pd.concat(frames).copy(deep=True)
result.head(900)