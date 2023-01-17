import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# ignore warnings

import warnings

warnings.filterwarnings("ignore")



from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import tree



import graphviz

from graphviz import Graph
df = pd.read_csv('../input/mushrooms.csv')
df.head()
df.rename(index=str, columns={'class':'eat_or_die'}, inplace=True)
df.head(40).T
df.shape
df.isnull().sum()
train, test = train_test_split(df, test_size=.3, random_state=123, stratify=df[['eat_or_die']])
train.shape
for column in train.columns:

    print('column = a', column)

    print(train.groupby(column)['eat_or_die'].value_counts())

    print()
df.dtypes
cat_df = pd.DataFrame()

for column in df.columns:

    cat_df[column] = df[column].astype('category')

cat_df.dtypes
for column in df.columns:

    cat_df[column] = cat_df[column].cat.codes

cat_df.head()
X = cat_df.drop(['eat_or_die'],axis=1)

y = cat_df[['eat_or_die']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123, stratify = df[['eat_or_die']])
X_train.shape
X_test.shape
y_train.shape
y_test.shape
X_train.shape[0]/(X_train.shape[0]+X_test.shape[0])
X_train.columns
y_train.columns
# for classificaiton you can change the algorithm as gini or entropy (information gain).  

# Default is gini.

clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=123)
clf = clf.fit(X_train, y_train)

clf
y_pred = clf.predict(X_train)

y_pred[0:5]
y_pred_proba = clf.predict_proba(X_train)

y_pred_proba
print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))
confusion_matrix(y_train, y_pred)
# labels = sorted(y_train.eat_or_die.unique())

# pd.DataFrame(confusion_matrix(y_train, y_pred), index=labels, columns=labels)



cm = pd.DataFrame(confusion_matrix(y_train, y_pred),

        columns=['Pred -', 'Pred +'], 

        index=['Actual -', 'Actual +'])



cm
tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()

print('true negatives: ', tn)

print('true positives: ', tp)

print('false positives (not as bad): ', fp)

print('false negatives (really bad to miss identifying a mushroom as being poisonous): ', fn)
print(classification_report(y_train, y_pred))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
# dot_data = tree.export_graphviz(clf, out_file=None) 

# graph = graphviz.Source(dot_data) 



# graph.render('mushroom_decision_tree', view=True)
df.odor.sort_values().unique()
cat_df.odor.sort_values().unique()
df[df.odor == 'n'].head()
cat_df[cat_df.odor == 5].head()
no_odor = cat_df[cat_df.odor == 5]

no_odor.head()
odor = cat_df[cat_df.odor != 5]

odor.head()
X = no_odor.drop(['eat_or_die'],axis=1)

y = no_odor[['eat_or_die']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123, 

                                                    stratify = no_odor[['eat_or_die']])



clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=123)



clf = clf.fit(X_train, y_train)



y_pred = clf.predict(X_train)



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))



cm = pd.DataFrame(confusion_matrix(y_train, y_pred),

        columns=['Pred -', 'Pred +'], 

        index=['Actual -', 'Actual +'])



print()

print(cm)

print()



tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()

print('true negatives: ', tn)

print('true positives: ', tp)

print('false positives (not as bad): ', fp)

print('false negatives (really bad to miss identifying a mushroom as being poisonous): ', fn)

print()



print(classification_report(y_train, y_pred))



print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
X = odor.drop(['eat_or_die'],axis=1)

y = odor[['eat_or_die']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123, 

                                                    stratify = odor[['eat_or_die']])



clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=123)



clf = clf.fit(X_train, y_train)



y_pred = clf.predict(X_train)



y_pred_proba = clf.predict_proba(X_train)



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))



cm = pd.DataFrame(confusion_matrix(y_train, y_pred),

        columns=['Pred -', 'Pred +'], 

        index=['Actual -', 'Actual +'])



print()

print(cm)

print()



tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()

print('true negatives: ', tn)

print('true positives: ', tp)

print('false positives (not as bad): ', fp)

print('false negatives (really bad to miss identifying a mushroom as being poisonous): ', fn)

print()



print(classification_report(y_train, y_pred))



print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
die_no_odor = no_odor[['odor', 'gill-color', 'eat_or_die']][no_odor.eat_or_die == 1]

die_no_odor.head()
die_no_odor.groupby('gill-color')['eat_or_die'].value_counts()
eat_no_odor = no_odor[['odor', 'gill-color', 'eat_or_die']][no_odor.eat_or_die == 0]

eat_no_odor.groupby('gill-color')['eat_or_die'].value_counts()
die_no_odor = no_odor[['odor', 'gill-color', 'ring-type', 'eat_or_die']][no_odor.eat_or_die == 1]

die_no_odor.head()
prob_colors = cat_df[cat_df['gill-color'].isin([2, 8, 10, 11])]

prob_colors.head(5)
out = (prob_colors.groupby(['gill-color', 'ring-type', 'stalk-color-below-ring'])

          ['eat_or_die'].value_counts()

          .rename('count').reset_index())

out

# out.loc[out.eat_or_die.eq(1)]
out = (prob_colors.groupby(['gill-color', 'ring-type', 'stalk-color-above-ring'])

          ['eat_or_die'].value_counts()

          .rename('count').reset_index())

out
pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', -1)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

out = (prob_colors.groupby(['gill-color', 'ring-type', 'stalk-color-below-ring', 

                      'stalk-color-above-ring', 'population', 'spore-print-color', 'stalk-root', 

                      'veil-color', 'habitat', 'cap-shape', 'ring-number', 'cap-color', 

                      'cap-surface'])

          ['eat_or_die'].value_counts()

          .rename('count').reset_index())



out
df['spore-print-color'].value_counts()
cat_df['spore-print-color'].value_counts()
no_odor['spore-print-color'].value_counts()
green_or_white = [5, 7]

gr_or_wh = no_odor[no_odor['spore-print-color'].isin(green_or_white)]

gr_or_wh.head()
gr_or_wh.shape
X = gr_or_wh.drop(['eat_or_die'],axis=1)

y = gr_or_wh[['eat_or_die']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123, 

                                                    stratify = gr_or_wh[['eat_or_die']])



clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=123)



clf = clf.fit(X_train, y_train)



y_pred = clf.predict(X_train)



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))



cm = pd.DataFrame(confusion_matrix(y_train, y_pred),

        columns=['Pred -', 'Pred +'], 

        index=['Actual -', 'Actual +'])



print()

print(cm)

print()



tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()

print('true negatives: ', tn)

print('true positives: ', tp)

print('false positives (not as bad): ', fp)

print('false negatives (really bad to miss identifying a mushroom as being poisonous): ', fn)

print()



print(classification_report(y_train, y_pred))



print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
not_gr_or_wh = no_odor[~no_odor['spore-print-color'].isin(green_or_white)]

not_gr_or_wh.head()
X = not_gr_or_wh.drop(['eat_or_die'],axis=1)

y = not_gr_or_wh[['eat_or_die']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123, 

                                                    stratify = not_gr_or_wh[['eat_or_die']])



clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=123)



clf = clf.fit(X_train, y_train)



y_pred = clf.predict(X_train)



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))



print(classification_report(y_train, y_pred))



print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
gr_or_wh.head()
bad_gr_or_wh = gr_or_wh[gr_or_wh['eat_or_die'] == 1]

out = (bad_gr_or_wh.groupby(['gill-color', 'ring-type', 'spore-print-color', 'population'])

          ['eat_or_die'].value_counts()

          .rename('count').reset_index())



out
good_gr_or_wh = gr_or_wh[gr_or_wh['eat_or_die'] == 0]

out = (good_gr_or_wh.groupby(['gill-color', 'ring-type', 'spore-print-color', 'population'])

          ['eat_or_die'].value_counts()

          .rename('count').reset_index())



out
clustered_or_several = [1, 4]

clu_or_sev = gr_or_wh[gr_or_wh['population'].isin(clustered_or_several)]

clu_or_sev.head()
X = clu_or_sev.drop(['eat_or_die'],axis=1)

y = clu_or_sev[['eat_or_die']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123, 

                                                    stratify = clu_or_sev[['eat_or_die']])



clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=123)



clf = clf.fit(X_train, y_train)



y_pred = clf.predict(X_train)



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))



cm = pd.DataFrame(confusion_matrix(y_train, y_pred),

        columns=['Pred -', 'Pred +'], 

        index=['Actual -', 'Actual +'])



print()

print(cm)

print()



tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()

print('true negatives: ', tn)

print('true positives: ', tp)

print('false positives (not as bad): ', fp)

print('false negatives (really bad to miss identifying a mushroom as being poisonous): ', fn)

print()



print(classification_report(y_train, y_pred))



print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
not_clu_or_sev = gr_or_wh[~gr_or_wh['population'].isin(clustered_or_several)]

not_clu_or_sev.head()
X = not_clu_or_sev.drop(['eat_or_die'],axis=1)

y = not_clu_or_sev[['eat_or_die']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123, 

                                                    stratify = not_clu_or_sev[['eat_or_die']])



clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=123)



clf = clf.fit(X_train, y_train)



y_pred = clf.predict(X_train)



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))



print(classification_report(y_train, y_pred))



print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))