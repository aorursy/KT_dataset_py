import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import scikitplot as skplt
# load the train and test data files

train_clean_standarized = pd.read_csv("../input/feature-exploration-and-dataset-preparation/train_clean_standarized.csv", index_col=0)

train_resampled_PCA = pd.read_csv("../input/pca-principal-component-analysis/train_PCA.csv", index_col=0)

train_resampled = pd.read_csv("../input/resampling/train_resampled.csv", index_col=0)

test = pd.read_csv("../input/santander-customer-satisfaction/test.csv", index_col=0)
# get our train test split data (25% test data)

y = train_clean_standarized.TARGET

X = train_clean_standarized.drop("TARGET", axis=1)

data_train, data_test, target_train, target_test = train_test_split(X, y, test_size=0.25, random_state=42)
# instantiate and fit the base model

# we're just picking some random hyperparameters

tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=50) 

tree_clf.fit(data_train, target_train)
fea_imp = pd.DataFrame({'imp': tree_clf.feature_importances_, 'col': X.columns})

fea_imp = fea_imp[fea_imp.imp > .005].sort_values(['imp', 'col'], ascending=[True, False])

fea_imp.plot(kind='barh', x='col', y='imp', legend=None)

plt.title('Decision Tree - Feature importance')

plt.ylabel('Features')

plt.xlabel('Importance');
# calculate the test set predictions

pred = tree_clf.predict(data_test)

skplt.metrics.plot_confusion_matrix(target_test, pred);
print(classification_report(target_test, pred))
# get our train test split data (25% test data)

y = train_resampled.TARGET

X = train_resampled.drop("TARGET", axis=1)



# we use stratify to balance our data sets

data_train, data_test, target_train, target_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)



# we assign a different weigth to the unsatisfied customer class (6.6 as there are 6.6 more satisfied customers in the dataset)

tree_clf_resampled = DecisionTreeClassifier(criterion='gini', max_depth=50, class_weight={0:1,1:6.6}) 

tree_clf_resampled.fit(data_train, target_train)



# calculate the test set predictions and display the confusion matrix and classification report

pred = tree_clf_resampled.predict(data_test)

skplt.metrics.plot_confusion_matrix(target_test, pred)

print(classification_report(target_test, pred))
# get our train test split data (25% test data)

y = train_resampled.TARGET

X = train_resampled.drop("TARGET", axis=1)



# we use stratify to balance our data sets

data_train, data_test, target_train, target_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)



# we assign a different weigth to the unsatisfied customer class (6.6 as there are 6.6 more satisfied customers in the dataset)

forest_clf = RandomForestClassifier(n_estimators=100, max_depth=50, class_weight={0:1,1:6.6}) 

forest_clf.fit(data_train, target_train)



# calculate the test set predictions and display the confusion matrix and classification report

pred = forest_clf.predict(data_test)

skplt.metrics.plot_confusion_matrix(target_test, pred)

print(classification_report(target_test, pred))
rf_param_grid = {

    'class_weight': [{0:1,1:3}, {0:1,1:6.6}, {0:1,1:10}],

    'max_depth': [None, 50, 100],

    'min_samples_split': [2, 40, 60],

    'min_samples_leaf': [1, 4]

}
rf_grid_search = GridSearchCV(RandomForestClassifier(n_estimators=100), rf_param_grid, cv=3, return_train_score=True)
rf_grid_search.fit(data_train, target_train)
rf_grid_search.best_params_
# calculate the test set predictions and display the confusion matrix and classification report

pred = rf_grid_search.best_estimator_.predict(data_test)

skplt.metrics.plot_confusion_matrix(target_test, pred)

print(classification_report(target_test, pred))
# get our train test split data (25% test data)

y = train_resampled_PCA.TARGET

X = train_resampled_PCA.drop("TARGET", axis=1)



# we use stratify to balance our data sets

data_train, data_test, target_train, target_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)



forest_clf_PCA = RandomForestClassifier(n_estimators=100, max_depth=50, class_weight={0:1,1:25}) 

forest_clf_PCA.fit(data_train, target_train)



# calculate the test set predictions and display the confusion matrix and classification report

pred = forest_clf_PCA.predict(data_test)

skplt.metrics.plot_confusion_matrix(target_test, pred)

print(classification_report(target_test, pred))
# prepare submission test data

column_diff = np.setdiff1d(test.columns.values, train_resampled.columns.values)

test_clean = test.drop(column_diff, axis=1)
# Decision Tree predictions

pred = tree_clf.predict(test_clean)

submission = pd.DataFrame({"ID":test_clean.index, "TARGET":pred})

#submission.to_csv("submission_DecisionTree.csv", index=False)

submission.TARGET.value_counts(0)
# Decision Tree predictions (resampled data)

pred = tree_clf_resampled.predict(test_clean)

submission = pd.DataFrame({"ID":test_clean.index, "TARGET":pred})

#submission.to_csv("submission_DecisionTree_Resampled.csv", index=False)

submission.TARGET.value_counts(0)
# Random Forest predictions

pred = forest_clf.predict(test_clean)

submission = pd.DataFrame({"ID":test_clean.index, "TARGET":pred})

#submission.to_csv("submission_RandomFores.csv", index=False)

submission.TARGET.value_counts(0)
# Grid Search Random Forest predictions

pred = rf_grid_search.best_estimator_.predict(test_clean)

submission = pd.DataFrame({"ID":test_clean.index, "TARGET":pred})

#submission.to_csv("submission_GridSearchRandomForest.csv", index=False)

submission.TARGET.value_counts(0)