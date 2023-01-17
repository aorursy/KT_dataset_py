!pip install --upgrade scikit-learn
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn import metrics

import pprint

import seaborn as sb

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from sklearn.inspection import permutation_importance # scikit-learn >=0.22 requires custom install



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def preprocess(d):

    binary_cols = ["surgery", "surgical_lesion", "cp_data", ]

    d[binary_cols] = d[binary_cols].replace({"yes":1,"no":0})

    d["age"] = d["age"].replace({"young":0, "adult":1})

    d["capillary_refill_time"] = d["capillary_refill_time"].replace({1:0, 2:1})

    d["outcome"] = d["outcome"].replace({"died":0, "euthanized":1, "lived": 2})

    

    # One hot encode variables

    d = pd.get_dummies(d, dummy_na=True)

    

    # Drop those where we don't know the outcome

    d = d[d.outcome.isnull()==False]

    

    # Rename outcome to torget (just to make it clearer)

    d["target"] = d["outcome"]

    del d["outcome"]



    y = d["target"]

    del d["target"]

    X = d

    

    # Split data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    

    # Replace na's - do it WITHOUT using the test data or we're cheating

    # Add isnull bool column to signal the value was actually unknown

    # Median - these cols are more or less normally distributed/have a bell shape

    nan_cols = ["packed_cell_volume", "pulse", "rectal_temp"]

    for c in nan_cols:

        med = X_train[c].median()

        X_train[f"{c}_nan"] = X_train[c].isnull().astype("int")

        X_train.loc[:, c] = X_train[c].fillna(med)

        X_test[f"{c}_nan"] = X_test[c].isnull().astype("int")

        X_test.loc[:, c] = X_test[c].fillna(med)

    

    # Mode - not normally distributed, just use the most frequent value

    nan_cols = ["respiratory_rate","nasogastric_reflux_ph", 

                "abdomo_protein", "total_protein"]

    

    for c in nan_cols:

        med = X_train[c].mode(dropna=True)[0]

        X_train[f"{c}_nan"] = X_train[c].isnull().astype("int")

        X_train.loc[:, c] = X_train[c].fillna(med)

        X_test[f"{c}_nan"] = X_test[c].isnull().astype("int")

        X_test.loc[:, c] = X_test[c].fillna(med)

    

    d["target"] = y

    return X_train, X_test, y_train, y_test, d
df = pd.read_csv('/kaggle/input/horse-colic/horse.csv')

X_train, X_test, y_train, y_test, df = preprocess(df)

X_train.head()
# Ensure no remaining nan's

X_train.isnull().sum().sum() #sort_values(ascending=False)
# Good candidates for feature selection

# Get correlated features

corred_cols = list(df.corr().target.abs().sort_values(ascending=False).head(n=30).index)

corred_cols = corred_cols[1:]

#df.corr().target.abs().sort_values(ascending=False).head(n=30)#.index

corred_cols
clf = RandomForestClassifier(random_state=0, n_estimators=100)

clf.fit(X_train[corred_cols], y_train)

#

# Use the forest's predict method on the test data

y_pred = clf.predict(X_test[corred_cols])

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Accuracy: %s' % metrics.accuracy_score(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred, digits=3))

cm = metrics.confusion_matrix(y_pred, y_test)

sb.set(font_scale=1.3)

sb.heatmap(cm, annot=True)

plt.show()
#pprint.pprint(dict(zip(corred_cols, clf.feature_importances_)))



importances = clf.feature_importances_

feat_importances = pd.Series(clf.feature_importances_, index=X_train[corred_cols].columns).sort_values(ascending=False)

feat_importances.nlargest(20).plot(kind='barh')
result = permutation_importance(clf, X_test[corred_cols], y_test, n_repeats=10,

                                random_state=42, n_jobs=2)

sorted_idx = result.importances_mean.argsort()



fig, ax = plt.subplots(figsize=(20,10))

ax.boxplot(result.importances[sorted_idx].T,

           vert=False, labels=X_test[corred_cols].columns[sorted_idx])

ax.set_title("Permutation Importances (test set)")

fig.tight_layout()

plt.show()
# Remove pulse

cols = corred_cols.copy()

cols.remove("pulse")

#cols.remove("peripheral_pulse_normal")

cols.remove("nasogastric_reflux_ph")

clf = RandomForestClassifier(random_state=42, n_estimators=100)

clf.fit(X_train[cols], y_train)

#

# Use the forest's predict method on the test data

y_pred = clf.predict(X_test[cols])

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Accuracy: %s' % metrics.accuracy_score(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred, digits=3))

cm = metrics.confusion_matrix(y_pred, y_test)

sb.set(font_scale=1.3)

sb.heatmap(cm, annot=True)

plt.show()
#x_valid, x_test, y_valid, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=1)

clf = lgb.LGBMClassifier(device='cpu',  

                         num_threads=6,

                         bagging_freq=5,

                         bagging_fraction= 0.9,

                         feature_fraction= 0.80,

                         learning_rate=0.05,

                         min_data_in_leaf=2,

                         num_leaves=81,

                         random_state=42

                        )





clf.fit(X_train[corred_cols], y_train)

y_pred = clf.predict(X_test[corred_cols])

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Accuracy: %s' % metrics.accuracy_score(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred, digits=3))

cm = metrics.confusion_matrix(y_pred, y_test)

sb.set(font_scale=1.3)

sb.heatmap(cm, annot=True)

plt.show()
# Use cols (no pulse)

clf = lgb.LGBMClassifier(device='cpu',  

                         num_threads=6,

                         bagging_freq=5,

                         bagging_fraction= 0.7, # changed

                         feature_fraction= 0.80,

                         learning_rate=0.05,

                         min_data_in_leaf=4, # changed

                         #num_leaves=81

                         random_state=42

                        )





clf.fit(X_train[cols], y_train)

y_pred = clf.predict(X_test[cols])

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Accuracy: %s' % metrics.accuracy_score(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred, digits=3))

cm = metrics.confusion_matrix(y_pred, y_test)

sb.set(font_scale=1.3)

sb.heatmap(cm, annot=True)

plt.show()