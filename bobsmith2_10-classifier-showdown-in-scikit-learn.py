import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



def warn(*args, **kwargs): pass

import warnings

warnings.warn = warn



from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit

from keras.utils import np_utils



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Swiss army knife function to organize the data



def encode(train, test):

    le = LabelEncoder().fit(train.species) 

    labels = le.transform(train.species)           # encode species strings

    classes = list(le.classes_)                    # save column names for submission

    test_ids = test.id                             # save test ids for submission

    

    train = train.drop(['species', 'id'], axis=1)  

    test = test.drop(['id'], axis=1)

    

    return train, labels, test, test_ids, classes



train, labels, test, test_ids, classes = encode(train, test)

train.head(1)
sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)



for train_index, test_index in sss:

    X_train, X_test = train.values[train_index], train.values[test_index]

    y_train, y_test = labels[train_index], labels[test_index]
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(1),

    KNeighborsClassifier(2),

    KNeighborsClassifier(3),

    KNeighborsClassifier(4),

    KNeighborsClassifier(5),

    KNeighborsClassifier(6),

    RandomForestClassifier(n_estimators=5),

    RandomForestClassifier(n_estimators=10),

    RandomForestClassifier(n_estimators=20),

    RandomForestClassifier(n_estimators=30),

    RandomForestClassifier(n_estimators=35),

    RandomForestClassifier(n_estimators=40),

    RandomForestClassifier(n_estimators=45),

    RandomForestClassifier(n_estimators=49),

    RandomForestClassifier(n_estimators=50),

    RandomForestClassifier(n_estimators=51),

    RandomForestClassifier(n_estimators=55),

    RandomForestClassifier(n_estimators=60),

    RandomForestClassifier(n_estimators=100),

    LinearDiscriminantAnalysis(solver='svd'),

    LinearDiscriminantAnalysis(solver='lsqr'),

    LinearDiscriminantAnalysis(solver='eigen')]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')

plt.show()



sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")



plt.xlabel('Log Loss')

plt.title('Classifier Log Loss')

plt.show()
# Predict Train Set

starting_clf = LinearDiscriminantAnalysis(solver='svd')

starting_clf.fit(X_train, y_train)

X_train_predictions = starting_clf.predict(X_train)

#X_train_predictions = starting_clf.predict_proba(X_train)



# Annotating Train Set

#print(X_train.shape)

#print(X_train_predictions.shape)

#z_train = np.concatenate((X_train, X_train_predictions), axis=1)

#print(z_train.shape)



# Annotating Test Set

#z_test_predictions = starting_clf.predict_proba(X_test)

#z_test = np.concatenate((X_test, z_test_predictions), axis=1)



print(y_train.shape)

print('='*5)

print(X_train_predictions.shape)

print('M'*5)

print(y_train[0])

print('W'*5)



z_train = X_train_predictions

for i in range(len(X_train_predictions)):

    if (X_train_predictions[i] == y_train[i]):

        z_train[i] = -1



#y_train_oh = np_utils.to_categorical(y_train, 99)

#z_train_proba = np.subtract(y_train_oh, X_train_predictions)

#X_test_predictions = starting_clf.predict_proba(X_test)



#for i in range(len(z_train)):

#if (z_train_proba.shape[-1] > 1):

#z_train = z_train_proba.argmax(axis=-1)

#else:

#    z_train = (z_train_proba > 0.5).astype('int32')



#print(type(z_train))

print('W'*5)

print(z_train.shape)

print(z_train[0:5])

print(y_train.shape)

print(z_train > -1)

#print(z_train_proba.shape[-1] > 1)
# Testing classifiers on the residual



for clf in classifiers:

    clf.fit(X_train, z_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions0 = starting_clf.predict(X_test)

    train_predictions1 = clf.predict(X_test)

    train_predictions0_ll = starting_clf.predict_proba(X_test)

    train_predictions1_ll = clf.predict_proba(X_test)



    print(train_predictions0.shape)

    print(train_predictions1.shape)

    print(train_predictions0_ll.shape)

    print(train_predictions1_ll.shape)



    train_predictions_ll = train_predictions1_ll

    

    for j in range(len(train_predictions1)):

        if (train_predictions1[j] == -1):

            train_predictions[j] = train_predictions0[j]

            train_predictions_ll[j] = train_predictions0_ll[j]

        else:

            train_predictions[j] = train_predictions1[j]

            train_predictions_ll[j] = train_predictions1_ll[j]



#    train_predictions = np.add(train_predictions0, train_predictions1)

#    if (train_predictions.shape[-1] > 1):

#        z_test = train_predictions.argmax

#    else:

#        z_test = (train_predictions > 0.5).astype('int32')

#    acc = accuracy_score(y_test, z_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    

    ll = log_loss(y_test, train_predictions_ll)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
# Predict Test Set

favorite_clf = LinearDiscriminantAnalysis()

favorite_clf.fit(X_train, y_train)

test_predictions = favorite_clf.predict_proba(test)



# Format DataFrame

submission = pd.DataFrame(test_predictions, columns=classes)

submission.insert(0, 'id', test_ids)

submission.reset_index()



# Export Submission

#submission.to_csv('submission.csv', index = False)

submission.tail()