# For handling DataFrames and NumPy arrays

import pandas as pd

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Put imports for the libraries you will use here

from sklearn.ensemble import RandomForestClassifier

# End of user imports



# Render and display charts in the notebook

%matplotlib inline

# Set a white theme for seaborn charts

sns.set_style('white')
df_train = pd.read_csv('../input/dsc-summer-school-random-forest-challenge/train.csv')

df_test = pd.read_csv('../input/dsc-summer-school-random-forest-challenge/test.csv')

df_sample = pd.read_csv('../input/dsc-summer-school-random-forest-challenge/sample_submission.csv')

df_sample
y_test = df_sample['is_edible']

len(y_test)
X_train = df_train.drop(['is_edible', 'id'], axis=1)

X_test = df_test.drop('id', axis=1)



y_train = df_train['is_edible']

y_test = df_sample['is_edible']

len(y_train)
X_train_ohe = pd.get_dummies(X_train)

X_test_ohe = pd.get_dummies(X_test)
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train_ohe, y_train)
y_pred = clf.predict(X_test_ohe)
len(y_pred)
len(y_train)
from sklearn.metrics import classification_report, confusion_matrix



def plot_confusion_matrix(cf_matrix):

    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = [f"{value:0.0f}" for value in cf_matrix.flatten()]

    group_percentages = [f"{value:.2%}" for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)



    fig = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return fig



print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred))
submission = pd.DataFrame(columns=['id', 'is_edible'])

submission['id'] = df_test['id']

submission['is_edible'] = y_pred

submission.to_csv('submission.csv', index=False)
# For handling DataFrames and NumPy arrays

import pandas as pd

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Render and display charts in the notebook

%matplotlib inline

# Set a white theme for seaborn charts

sns.set_style('white')
df_train = pd.read_csv('../input/dsc-summer-school-random-forest-challenge/train.csv')

df_test = pd.read_csv('../input/dsc-summer-school-random-forest-challenge/test.csv')

df_sample = pd.read_csv('../input/dsc-summer-school-random-forest-challenge/sample_submission.csv')
X_train = df_train.drop(['is_edible', 'id'], axis=1)

X_test = df_test.drop('id', axis=1)



y_train = df_train['is_edible']

y_test = df_sample['is_edible']
X_train_ohe = pd.get_dummies(X_train)

X_test_ohe = pd.get_dummies(X_test)
clf = RandomForestClassifier(n_jobs=-1)

clf.fit(X_train_ohe, y_train)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(random_state=42, n_jobs=-1)

clf.fit(X_train_ohe, y_train)
y_pred = clf.predict(X_test_ohe)
from sklearn.metrics import classification_report, confusion_matrix



def plot_confusion_matrix(cf_matrix):

    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = [f"{value:0.0f}" for value in cf_matrix.flatten()]

    group_percentages = [f"{value:.2%}" for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)



    fig = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return fig



print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred))
# If you want to use Grid search, change to GridSearchCV. Don't forget to import first!

# But be warned, it will take a long time

# param_search = RandomizedSearchCV(estimator=clf, # Your model variable goes here,

#                                   param_distributions=param_grid, # Your parameter search space

#                                   n_iter=100, # How many times do you want it to run?

#                                   cv=5,

#                                   verbose=2, # Print messages. Set to 0 to silence it

#                                   n_jobs=-1) # Use all available CPU cores