#22p21c0878_ณัฐภัทร_W2HW2_27092020



import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import precision_score, recall_score, f1_score

from tabulate import tabulate
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]



X = pd.get_dummies(train_data[features])

X["Age"] = X["Age"].fillna(X["Age"].mean())



X_test = pd.get_dummies(test_data[features])

X_test["Age"] = X_test["Age"].fillna(X["Age"].mean())



derror = 1.1

gerror = 1.1

nerror = 1.1



dtcResult = []

gNBResult = []

nnResult = []



def decisionTreeClassifier(trainfeat, trainlabel, testfeat, testlabel, number):

    global derror, dindex

    tree = DecisionTreeClassifier(random_state=0)

    tree.fit(trainfeat, trainlabel)

    predict = tree.predict(testfeat)

    precision = precision_score(testlabel, predict)

    recall = recall_score(testlabel, predict)

    f1_score = 2 * (precision * recall) / (precision + recall)

    dtcResult.append(['class'+str(number+1), recall, precision, f1_score])

    if derror > sum([1 for i in range(predict.size) if predict[i]!=testlabel[i]])/predict.size:

        derror = sum([1 for i in range(predict.size) if predict[i]!=testlabel[i]])/predict.size

        dindex = number



def gaussianNB(trainfeat, trainlabel, testfeat, testlabel, number):

    global gerror, gindex

    model = GaussianNB()

    model.fit(trainfeat, trainlabel)

    predict = model.predict(testfeat)

    precision = precision_score(testlabel, predict)

    recall = recall_score(testlabel, predict)

    f1_score = 2 * (precision * recall) / (precision + recall)

    gNBResult.append(['class'+str(number+1), recall, precision, f1_score])

    if gerror > sum([1 for i in range(predict.size) if predict[i]!=testlabel[i]])/predict.size:

        gerror = sum([1 for i in range(predict.size) if predict[i]!=testlabel[i]])/predict.size

        gindex = number



def neuralNetwork(trainfeat, trainlabel, testfeat, testlabel, number):

    global nerror, nindex

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(350, input_shape=X.shape, activation='relu'))

    model.add(tf.keras.layers.Dense(50, activation='relu'))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))

    model.fit(trainfeat, trainlabel, epochs=2000, verbose=0)

    predict = np.array([int(round(i[0])) for i in model.predict(testfeat)])

    precision = precision_score(testlabel, predict)

    recall = recall_score(testlabel, predict)

    f1_score = 2 * (precision * recall) / (precision + recall)

    nnResult.append(['class'+str(number+1), recall, precision, f1_score])

    if nerror > sum([1 for i in range(predict.size) if predict[i]!=testlabel[i]])/predict.size:

        nerror = sum([1 for i in range(predict.size) if predict[i]!=testlabel[i]])/predict.size

        nindex = number



def genFold(x):

    trainfeat = np.array([row for index, row in X.iterrows() if index%5 != x])

    trainlabel = np.array([row for index, row in enumerate(y) if index%5 != x])

    testfeat = np.array([row for index, row in X.iterrows() if index%5 == x])

    testlabel = np.array([row for index, row in enumerate(y) if index%5 == x])

    return trainfeat, trainlabel, testfeat, testlabel



for i in range(5):

    trainfeat, trainlabel, testfeat, testlabel = genFold(i)

    decisionTreeClassifier(trainfeat, trainlabel, testfeat, testlabel, i)

    gaussianNB(trainfeat, trainlabel, testfeat, testlabel, i)

    neuralNetwork(trainfeat, trainlabel, testfeat, testlabel, i)



# trainfeat, trainlabel, testfeat, testlabel = genFold(dindex)

# tree = DecisionTreeClassifier(random_state=0)

# tree.fit(trainfeat, trainlabel)

# predictions = tree.predict(X_test)



# trainfeat, trainlabel, testfeat, testlabel = genFold(gindex)

# model = GaussianNB()

# model.fit(trainfeat, trainlabel)

# predictions = model.predict(X_test)



# trainfeat, trainlabel, testfeat, testlabel = genFold(nindex)

# model = tf.keras.Sequential()

# model.add(tf.keras.layers.Dense(350, input_shape=X.shape, activation='relu'))

# model.add(tf.keras.layers.Dense(50, activation='relu'))

# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))

# model.fit(trainfeat, trainlabel, epochs=2000, verbose=0)

# predictions = [int(round(i[0])) for i in model.predict(X_test)]



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")



header = ['Class', 'Recall', 'Precision', 'F-Measure']

print("Decision Tree")

print(tabulate(dtcResult, headers=header))

print(f"Average F-Measure: {sum([res[3] for res in dtcResult])/5}")

print()

print("Naive Bayes")

print(tabulate(gNBResult, headers=header))

print(f"Average F-Measure: {sum([res[3] for res in gNBResult])/5}")

print()

print("Neural Network")

print(tabulate(nnResult, headers=header))

print(f"Average F-Measure: {sum([res[3] for res in nnResult])/5}")
