import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.decomposition import TruncatedSVD

from sklearn.manifold import TSNE



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv('../input/digit-recognizer/train.csv')

test_df = pd.read_csv('../input/digit-recognizer/test.csv')
fig = plt.figure()

plt.hist(train_df['label'],histtype='bar',rwidth=0.8)

fig.suptitle('Class distribution in Data',fontsize=15)

plt.xlabel('classes')

plt.ylabel('count')
features = [col for col in train_df.columns if col.startswith('pixel')]

X_train, X_val, y_train, y_val = train_test_split(train_df[features], 

                                                  train_df['label'], 

                                                  test_size=0.25)
clf = tree.DecisionTreeClassifier(max_depth=10, random_state=0)

clf.fit(X_train, y_train)
def acc(y_true, y_pred):

    return round(accuracy_score(y_true, y_pred) * 100, 2)
y_pred = clf.predict(X_val)

print(acc(y_val,y_pred))

print(acc(y_train,clf.predict(X_train)))
concat_df = pd.concat([train_df,test_df], sort=True)

del concat_df['label']

tsvd = TruncatedSVD(n_components=50).fit_transform(concat_df)
X_train, X_val, y_train, y_val = train_test_split(tsvd[:42000], train_df['label'], test_size=0.25, random_state=0)

clf = tree.DecisionTreeClassifier(max_depth=10, random_state=0)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

print(acc(y_val,y_pred))

print(acc(y_train,clf.predict(X_train)))
tsne = TSNE(n_components=2)

transformed = tsne.fit_transform(tsvd)
tsne_train = pd.DataFrame(transformed[:len(train_df)], columns=['component1', 'component2'])

tsne_test = pd.DataFrame(transformed[len(train_df):], columns=['component1', 'component2'])
X_train, X_val, y_train, y_val = train_test_split(tsne_train, 

                                                  train_df['label'], 

                                                  test_size=0.25, 

                                                  random_state=0)

clf = tree.DecisionTreeClassifier(max_depth=10, random_state=0)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

print(acc(y_val,y_pred))

print(acc(y_train,clf.predict(X_train)))
predictions = clf.predict(tsne_test)
# ImageId,Label



test_df['Label'] = pd.Series(predictions)

test_df['ImageId'] = test_df.index +1

sub = test_df[['ImageId','Label']]



sub.to_csv('submission.csv', index=False)