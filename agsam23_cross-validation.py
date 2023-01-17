import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import model_selection

from sklearn import metrics

from sklearn import tree
data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data.head()
print(data.info())
data.quality.unique()
map = {

    3: 0,

    4: 1,

    5: 2,

    6: 3,

    7: 4,

    8: 5

}

data.quality = data.quality.map(map)
data.quality.unique()
df_train = data.head(1000)

df_label = df_train['quality']

df_train = df_train.drop('quality', axis = 1)
df_train.head()
df_test = data.tail(599)

test_label = df_test['quality']

df_test= df_test.drop('quality', axis = 1)
train_acc = [0.5]

test_acc = [0.5]

for depth in range(1, 20):

    clf = tree.DecisionTreeClassifier(max_depth = depth)

    

    clf.fit(df_train, df_label)

    

    train_pred = clf.predict(df_train)

    acc_train = metrics.accuracy_score(df_label, train_pred)

    

    test_pred = clf.predict(df_test)

    acc_test = metrics.accuracy_score(test_label, test_pred)

    

    train_acc.append(acc_train)

    test_acc.append(acc_test)
plt.figure(figsize=(10,5))

sns.set_style('whitegrid')

plt.plot(train_acc, label = 'train accuracy')

plt.plot(test_acc, label = 'test accuracy')

plt.legend(loc='upper left', prop = {'size': 15})

plt.xticks(range(0, 20, 5))

plt.xlabel('max_depth', size = 20)

plt.ylabel('accuracy', size = 20)

plt.show()
#Applying KFold Cross validation

#create a new column kfold with entries -1

data['kfold'] = -1

#Shuffle data

data = data.sample(frac = 1).reset_index(drop = True)

#Split data into 5 folds

kf = model_selection.KFold(n_splits = 5)

for fold, (t, v) in enumerate(kf.split(X=data)):

    data.loc[v, 'kfold'] = fold

#Saving data for further use

data.to_csv('t_fold.csv', index = False)
def check(fold):

    df = pd.read_csv('./t_fold.csv')

    df_train = df[df.kfold != fold].reset_index(drop = True)

    df_test = df[df.kfold == fold].reset_index(drop = True)  

    

    y_train = df_train.quality.values

    x_train = df_train.drop('quality', axis = 1).values

    

    y_valid = df_test.quality.values

    x_valid = df_test.drop('quality', axis = 1).values

    

    ktrain_acc = [0.5]

    ktest_acc = [0.5]

    for depth in range(1, 20):

        clf = tree.DecisionTreeClassifier(max_depth = depth)



        clf.fit(x_train, y_train)



        train_pred = clf.predict(x_train)

        acc_train = metrics.accuracy_score(y_train, train_pred)



        test_pred = clf.predict(x_valid)

        acc_test = metrics.accuracy_score(y_valid, test_pred)



        ktrain_acc.append(acc_train)

        ktest_acc.append(acc_test)

    plt.figure(figsize=(10,5))

    sns.set_style('whitegrid')

    plt.plot(ktrain_acc, label = 'train accuracy')

    plt.plot(ktest_acc, label = 'test accuracy')

    plt.legend(loc='upper left', prop = {'size': 15})

    plt.xticks(range(0, 20, 5))

    plt.xlabel('max_depth', size = 20)

    plt.ylabel('accuracy', size = 20)

    plt.show()
#Taking fold 0 as test set and rest as training set

check(fold = 0)
#Taking fold 1 as test set and rest as training set

check(fold = 1)
#Taking fold 2 as test set and rest as training set

check(fold = 2)
#Taking fold 3 as test set and rest as training set

check(fold = 3)
#Taking fold 4 as test set and rest as training set

check(fold = 4)
X = pd.read_csv('./t_fold.csv')

y = X.quality.values

X = X.drop('quality', axis = 1).values

clf = tree.DecisionTreeClassifier(max_depth = 20)

scores = model_selection.cross_val_score(clf, X, y, cv=5)

print(scores)
#Calculating the average

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
sns.countplot(data['quality'])
#Applying stratied k-fold

y_data = data.quality.values

kf = model_selection.StratifiedKFold(n_splits = 5)

for fold, (t, v) in enumerate(kf.split(X=data, y = y_data)):

    data.loc[v, 'kfold'] = fold

data.to_csv('st_fold.csv', index = False)
def check_stratified(fold):

    df = pd.read_csv('./st_fold.csv')

    df_train = df[df.kfold != fold].reset_index(drop = True)

    df_test = df[df.kfold == fold].reset_index(drop = True)  

    

    y_train = df_train.quality.values

    x_train = df_train.drop('quality', axis = 1).values

    

    y_valid = df_test.quality.values

    x_valid = df_test.drop('quality', axis = 1).values

    

    sktrain_acc = [0.5]

    sktest_acc = [0.5]

    for depth in range(1, 20):

        clf = tree.DecisionTreeClassifier(max_depth = depth)



        clf.fit(x_train, y_train)



        train_pred = clf.predict(x_train)

        acc_train = metrics.accuracy_score(y_train, train_pred)



        test_pred = clf.predict(x_valid)

        acc_test = metrics.accuracy_score(y_valid, test_pred)



        sktrain_acc.append(acc_train)

        sktest_acc.append(acc_test)

    plt.figure(figsize=(10,5))

    sns.set_style('whitegrid')

    plt.plot(sktrain_acc, label = 'train accuracy')

    plt.plot(sktest_acc, label = 'test accuracy')

    plt.legend(loc='upper left', prop = {'size': 15})

    plt.xticks(range(0, 20, 5))

    plt.xlabel('max_depth', size = 20)

    plt.ylabel('accuracy', size = 20)

    plt.show()
#Taking fold 0 as test set and rest as training set

check_stratified(fold = 0)
#Taking fold 1 as test set and rest as training set

check_stratified(fold = 1)
#Taking fold 2 as test set and rest as training set

check_stratified(fold = 2)
#Taking fold 3 as test set and rest as training set

check_stratified(fold = 3)
#Taking fold 4 as test set and rest as training set

check_stratified(fold = 4)
X = pd.read_csv('./st_fold.csv')

y = X.quality.values

X = X.drop('quality', axis = 1).values

clf = tree.DecisionTreeClassifier(max_depth = 20)

scores = model_selection.cross_val_score(clf, X, y, cv=5)

print(scores)

#Calculating the average

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))