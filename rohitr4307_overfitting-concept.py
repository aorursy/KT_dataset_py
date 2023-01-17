import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import tree

from sklearn import metrics

from sklearn import model_selection



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data.head()
data.shape
quality_map = {

    3 : 0,

    4 : 1,

    5 : 2,

    6 : 3,

    7 : 4,

    8 : 5

}

data.loc[:, 'quality'] = data.quality.map(quality_map)
data = data.sample(frac=1).reset_index(drop=True)



df_train = data.head(1000)

df_test = data.tail(599)
columns = data.drop('quality', axis=1).columns.tolist()
# Decision Tree - max depth = 3



clf = tree.DecisionTreeClassifier(max_depth=3)

clf.fit(df_train[columns], df_train.quality)



train_prediction = clf.predict(df_train[columns])

test_prediction = clf.predict(df_test[columns])



train_accuracy = metrics.accuracy_score(df_train.quality, train_prediction)

test_accuracy = metrics.accuracy_score(df_test.quality, test_prediction)



print('Train accuracy for max depth 3 - ', format(train_accuracy*100))

print('Test accuracy for max depth 3 - ', format(test_accuracy*100))
# Decision Tree - max depth = 10



clf = tree.DecisionTreeClassifier(max_depth=10)

clf.fit(df_train[columns], df_train.quality)



train_prediction = clf.predict(df_train[columns])

test_prediction = clf.predict(df_test[columns])



train_accuracy = metrics.accuracy_score(df_train.quality, train_prediction)

test_accuracy = metrics.accuracy_score(df_test.quality, test_prediction)



print('Train accuracy for max depth 3 - ', format(train_accuracy*100))

print('Test accuracy for max depth 3 - ', format(test_accuracy*100))
train_accuracy_list = []

test_accuracy_list = []



for i in range(1, 25):

    clf = tree.DecisionTreeClassifier(max_depth=i)

    clf.fit(df_train[columns], df_train.quality)



    train_prediction = clf.predict(df_train[columns])

    test_prediction = clf.predict(df_test[columns])



    train_accuracy = metrics.accuracy_score(df_train.quality, train_prediction)

    test_accuracy = metrics.accuracy_score(df_test.quality, test_prediction)



    train_accuracy_list.append(train_accuracy)

    test_accuracy_list.append(test_accuracy)
# Plot Accuracy



plt.figure(figsize=(10, 5))

sns.set_style('whitegrid')

plt.plot(train_accuracy_list, label='Train Accuracy')

plt.plot(test_accuracy_list, label='Test Accuracy')

plt.legend(loc='upper left', prop={'size':15})

plt.xticks(range(0, 26, 5))

plt.xlabel('Max Depth', size=20)

plt.ylabel('Accuracy', size=20)

plt.show()
plt.figure(figsize=(12, 5))

sns.set_style('whitegrid')

sns.countplot(data.quality)

# plt.legend('upper-left', prop={'size':15})

plt.xlabel('Quality', size=20)

plt.ylabel('Count' ,size=20)

plt.title('Target - Quality', fontsize=22)

plt.show()
def kfold(df, target_col='quality', nfold=3):

    

    df['fold'] = -1

    

    X = df.drop(target_col, axis=1)

    Y = df[target_col]

    

    kf = model_selection.StratifiedKFold(n_splits=nfold)

    

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, Y)):

        

        df.loc[val_idx, 'fold'] = fold

    

    return df
fold_cnt = 4

data = kfold(data, 'quality', fold_cnt)
def train_model(df_train, df_test, fold, columns):

    train_accuracy_list = []

    test_accuracy_list = []



    for i in range(1, 25):

        clf = tree.DecisionTreeClassifier(max_depth=i)

        clf.fit(df_train[columns], df_train.quality)



        train_prediction = clf.predict(df_train[columns])

        test_prediction = clf.predict(df_test[columns])



        train_accuracy = metrics.accuracy_score(df_train.quality, train_prediction)

        test_accuracy = metrics.accuracy_score(df_test.quality, test_prediction)



        train_accuracy_list.append(train_accuracy)

        test_accuracy_list.append(test_accuracy)

    

    accuracy_graph(train_accuracy_list, test_accuracy_list, fold)

    

    
def accuracy_graph(train_accuracy_list, test_accuracy_list, fold):

    # Plot Accuracy



    plt.figure(figsize=(10, 5))

    sns.set_style('whitegrid')

    plt.plot(train_accuracy_list, label='Train Accuracy')

    plt.plot(test_accuracy_list, label='Test Accuracy')

    plt.legend(loc='upper left', prop={'size':15})

    plt.xticks(range(0, 26, 5))

    plt.xlabel('Max Depth', size=20)

    plt.ylabel('Accuracy', size=20)

    plt.title('Fold - {}'.format(fold), fontsize=22)

    plt.show()
for i in range(fold_cnt):

    df_train = data[data.fold != i]

    df_test = data[data.fold == i]

    train_model(df_train, df_test, i, columns)