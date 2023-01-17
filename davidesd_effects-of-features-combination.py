import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit



train = pd.read_csv('../input/../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Swiss army knife function to organize the data

def encode(train, test):

    le = LabelEncoder().fit(train.species) 

    labels = le.transform(train.species)           # encode species strings

    classes = list(le.classes_)                    # save column names for submission

    test_ids = test.id                             # save test ids for submission

    

    train = train.drop(['species', 'id'], axis=1)  

    test = test.drop(['id'], axis=1)

    

    cols_s = [c for c in train.columns if 'shape' in c]

    cols_t = [c for c in train.columns if 'texture' in c]

    cols_m = [c for c in train.columns if 'margin' in c]

    cols_st = [c for c in train.columns if ('shape' in c or 'texture' in c)]

    cols_mt = [c for c in train.columns if ('margin' in c or 'texture' in c)]

    cols_sm = [c for c in train.columns if ('margin' in c or 'shape' in c)]



    train_set = [train[cols_s], train[cols_m], train[cols_t], train[cols_st], train[cols_sm], train[cols_mt], train]

    test_set = [test[cols_s], test[cols_m], test[cols_t], test[cols_st], test[cols_sm], test[cols_mt], test]

    

    return train, labels, test, test_ids, classes, train_set, test_set



train, labels, test, test_ids, classes, train_set, test_set = encode(train, test)
# Simply looping through 10 out-of-the box classifiers and printing the results.

# Obviously, these will perform much better after tuning their hyperparameters, 

# but this gives you a decent ballpark idea

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]
# Logging for Visual Comparison

log_cols=["Classifier", "Acc Sh", "Acc Ma", "Acc Te", "Acc Sh-Te", "Acc Sh-Ma", "Acc Ma-Te", "Acc Sh-Ma-Te", "LL Sh", "LL Ma", "LL Te", "LL Sh-Te", "LL Sh-Ma", "LL Ma-Te", "LL Sh-Ma-Te"]

log = pd.DataFrame(columns=log_cols)

feat = ["Shape", "Margin", "Texture", "Shape-Texture", "Shape-Margin", "Margin-Texture", "Shape-Margin-Texture"]



for clf in classifiers:

    name = clf.__class__.__name__

    print("="*30)

    print(name)

    i = 0

    acc_col = []

    ll_col = []    

    

    for tr, te in zip(train_set, test_set):

        print("\t" + feat[i])

        i += 1

        # Stratification is necessary for this dataset because there is a relatively 

        # large number of classes (100 classes for 990 samples). This will ensure we 

        # have all classes represented in both the train and test indices

        sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

        

        for train_index, test_index in sss:

            X_train, X_test = tr.values[train_index], tr.values[test_index]

            y_train, y_test = labels[train_index], labels[test_index]

        

        # train the classifier

        clf.fit(X_train, y_train)

        

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        acc_col.append(acc)

        

        train_predictions = clf.predict_proba(X_test)

        ll = log_loss(y_test, train_predictions)

        ll_col.append(ll)

        

    log_entry = pd.DataFrame([[name]+acc_col+ll_col], columns=log_cols)

    log = log.append(log_entry)
cols = [c for c in log.columns if c[0:2] != 'LL']

df1 = log[cols]

cols = [c for c in log.columns if c[0:2] != 'Ac']

df2 = log[cols]

df1.plot(kind='barh', x='Classifier', title="Accuracy", figsize=(10,20), width=0.8, colormap="jet")

df2.plot(kind='barh', x='Classifier', title="Log Loss", figsize=(10,20), width=0.8, colormap="jet")