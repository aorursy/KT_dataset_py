import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import datetime 
import matplotlib.pyplot as plt
%matplotlib inline
pandas_train = pd.read_csv("../input/train.csv")
X_pandas = pandas_train.drop(["Cover_Type", "Id"], axis=1)
columns = pandas_train.axes[1]
X = np.array(X_pandas)
Y = np.array(pandas_train)[:,-1]
Y = Y.reshape([Y.shape[0], 1])
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)
acc_train, acc_valid = [], []
for i in range(3, 35):  
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=1)
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_train)
    acc_score_train = accuracy_score(Y_train, prediction)
    acc_train.append(acc_score_train)
    prediction = clf.predict(X_valid)
    acc_score_valid = accuracy_score(Y_valid, prediction)
    acc_valid.append(acc_score_valid)
    print("""Depth = {0}, Train accuracy = {1},
          Valid accuracy = {2}""".format(i, acc_score_train, acc_score_valid))
plt.plot(range(3, 35), acc_train)
plt.plot(range(3, 35), acc_valid)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('depth')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
acc_train, acc_valid = [], []    
for i in range(3, 35):  
    clf = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=1)
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_train)
    acc_score_train = accuracy_score(Y_train, prediction)
    acc_train.append(acc_score_train)
    prediction = clf.predict(X_valid)
    acc_score_valid = accuracy_score(Y_valid, prediction)
    acc_valid.append(acc_score_valid)
    print("""Depth = {0}, Train accuracy = {1}",
          Valid accuracy = {2}""".format(i, acc_score_train, acc_score_valid))  
plt.plot(range(3, 35), acc_train)
plt.plot(range(3, 35), acc_valid)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('depth')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
importances = clf.feature_importances_
zero_imp_columns = []
print("\nNot important features:")
for i, name in zip(importances, columns):
    if i == 0.0:
        print("{0}:{1}".format(name, i))
        zero_imp_columns.append(name)
for f in zero_imp_columns:
    pandas_train = pandas_train.drop([f], axis=1)

X_pandas = pandas_train.drop(["Cover_Type", "Id"], axis=1)
columns = pandas_train.axes[1]
X = np.array(X_pandas)
Y = np.array(pandas_train)[:,-1]
Y = Y.reshape([Y.shape[0], 1])
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)
start_time = datetime.datetime.now()
clf = DecisionTreeClassifier(criterion='entropy', max_depth=26, random_state=1)
clf.fit(X_train, Y_train)
prediction = clf.predict(X_train)
acc_score_train = accuracy_score(Y_train, prediction)
prediction = clf.predict(X_valid)
acc_score_valid = accuracy_score(Y_valid, prediction)
print("""\nDecision tree time execution(criterion='entropy' and depth=26 ): {0},
      Train accuracy = {1},
      Valid accuracy = {2}""".format(datetime.datetime.now() - start_time, acc_score_train, acc_score_valid))
start_time = datetime.datetime.now()
clf = DecisionTreeClassifier(criterion='gini', max_depth=31, random_state=1)
clf.fit(X_train, Y_train)
print("""\nDecision tree time execution(criterion='gini' and depth=31 ): {0},
      Train accuracy = {1},
      Valid accuracy = {2}""".format(datetime.datetime.now() - start_time, acc_score_train, acc_score_valid))