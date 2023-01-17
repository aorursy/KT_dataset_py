import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
training_set = pd.read_csv('/kaggle/input/unsw-nb15/UNSW_NB15_training-set.csv')

training_set.head()
training_set.info()
training_set.head()
mask = (training_set.dtypes == np.object)

print(training_set.loc[:,mask].head())

list_cat = training_set.loc[:,mask].columns.tolist()

print(list_cat)

print(training_set.loc[:,mask].values)
mask = (training_set.dtypes != np.object)

print(training_set.loc[:,mask].head())

list_cat = training_set.loc[:,mask].columns.tolist()

print(list_cat)

training_set.loc[:,mask].describe()
#  Check whether the positive label (1) match attack categories, and whether attack categories match labelled data.



# all(iterable) returns True if all elements of the iterable are considered as true values

print(all(((training_set.label == 1) & (training_set.attack_cat != 'Normal')) == (training_set.attack_cat != 'Normal')))

print(all(((training_set.attack_cat != 'Normal') & (training_set.label == 1)) == (training_set.label == 1)))
# number of occurrences for each attack category

training_set.attack_cat.value_counts()
mask = (training_set.label == 1)

print(training_set.loc[mask,:].service.value_counts())

print(training_set.loc[mask,:].proto.value_counts())
mask = (training_set.label == 0)

print(training_set.loc[mask,:].service.value_counts())

print(training_set.loc[mask,:].proto.value_counts())
Y = training_set.label

X = training_set.drop(columns=['id','attack_cat','label'])

mask = (X.dtypes == np.object)

list_cat = X.loc[:,mask].columns.tolist()

list_cat
X = pd.get_dummies(X, columns=list_cat)

X.head()
Y.head()
import sklearn

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
import xgboost as xgb

from sklearn.metrics import classification_report,roc_auc_score,average_precision_score
params = {

    'max_depth': 10,

    'objective': 'multi:softmax',  # error evaluation for multiclass training

    'num_class': 2,                # Number of classes 

    'n_gpus': 4

}



xg_clf = xgb.XGBClassifier(**params)

xg_clf.fit(X_train, y_train)
pred = xg_clf.fit(X_train, y_train).predict(X_test)

print(classification_report(y_test, pred))
roc_auc_score(y_test, pred)
print('AUPRC = {}'.format(average_precision_score(y_test,pred)))
## PLOT IMPORTANCE OF FEATURES with type cover

# ”cover” is the average coverage of splits which use the feature where coverage is defined as the number of samples affected by the split

xgb.plot_importance(xg_clf, importance_type='cover')

plt.rcParams['figure.figsize'] = [10, 20]

plt.show()
## PLOT IMPORTANCE OF FEATURES with type weight

# ”weight” is the number of times a feature appears in a tree

xgb.plot_importance(xg_clf, importance_type='weight')

plt.show()
## PLOT IMPORTANCE OF FEATURES with type gain

# ”gain” is the average gain of splits which use the feature

xgb.plot_importance(xg_clf, importance_type='gain')

plt.show()
# plot single tree

from xgboost import plot_tree

from matplotlib.pylab import rcParams

##set up the parameters

rcParams['figure.figsize'] = 30,50

print('This is a plot of the first decision tree in the model (index 0), showing the features and feature values for each split as well as the output leaf nodes.!')

plot_tree(xg_clf, num_trees=0, rankdir='LR')

plt.show()