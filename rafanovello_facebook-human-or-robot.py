import itertools

from collections import defaultdict



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.svm import SVC

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



%matplotlib inline
train = pd.read_csv('../input/train.csv')

train.head()
bids = pd.read_csv('../input/bids.csv')

bids.head()
train_bids = pd.merge(bids, train, how='left', on='bidder_id')

train_bids.head()
train_bids.dropna(subset=['outcome'], inplace=True)

train_bids.dropna(subset=['country'], inplace=True)

train_bids.head()
del train

del bids
to_plot = train_bids.groupby('outcome')['bid_id'].count()

to_plot = to_plot.groupby(level=0).apply(lambda x: 100 * x / train_bids.shape[0]).reset_index()

print(to_plot)

ax = sns.barplot(x="outcome", y="bid_id", data=to_plot);

ax.set(ylabel="Percent");
to_plot = train_bids.groupby(['outcome', 'merchandise'])['bid_id'].count()
fig, ax = plt.subplots(figsize=(15, 7))

sns.barplot(x='merchandise', y='bid_id', hue='outcome', data=to_plot.reset_index(), ax=ax);
to_plot = train_bids.groupby(['outcome', 'country'])['bid_id'].count()

fig, ax = plt.subplots(figsize=(15, 7))

sns.barplot(x='country', y='bid_id', hue='outcome', data=to_plot.reset_index(), ax=ax);
X = train_bids.copy()

X.drop(['bid_id', 'outcome', 'time'], axis=1, inplace=True)

X = X[['bidder_id', 'payment_account', 'address']]



d = defaultdict(LabelEncoder)

X = X.apply(lambda x: d[x.name].fit_transform(x))

X.head()
y = np.ravel(train_bids.outcome)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
forest = ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=4)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

indices = np.argsort(importances)[::-1]

feature_names = [X.columns[i] for i in indices]



plt.figure(figsize=(15,6))

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), feature_names)

plt.xlim([-1, X.shape[1]])

plt.show()
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.4f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
class_names = ['human', 'robot']

y_pred = forest.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=4)



plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Confusion matrix, with normalization')

precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
test = pd.read_csv('../input/test.csv')

test.head()
d = defaultdict(LabelEncoder)

test = test.apply(lambda x: d[x.name].fit_transform(x))



y_pred = forest.predict(test)
predic = pd.Series(y_pred)

predic.head()
test = test.apply(lambda x: d[x.name].inverse_transform(x))

test['prediction'] = predic

test.head()
test[['bidder_id', 'prediction']].to_csv('submission.csv', index=False)
test[test.bidder_id == 'eaf0ed0afc9689779417274b4791726cn5udi']