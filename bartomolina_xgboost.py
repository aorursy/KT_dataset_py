import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
# load the train and test data files

train_clean_standarized = pd.read_csv("../input/feature-exploration-and-dataset-preparation/train_clean_standarized.csv", index_col=0)

test = pd.read_csv("../input/santander-customer-satisfaction/test.csv", index_col=0)
# get our train test split data (25% test data)

y = train_clean_standarized.TARGET

X = train_clean_standarized.drop("TARGET", axis=1)

data_train, data_test, target_train, target_test = train_test_split(X, y, test_size=0.25, random_state=42)
# instantiate and fit XGBClassifier

clf = XGBClassifier()

clf.fit(data_train, target_train)



# predict on training and test sets

training_preds = clf.predict(data_train)

test_preds = clf.predict(data_test)



# accuracy of training and test sets

training_accuracy = accuracy_score(target_train, training_preds)

test_accuracy = accuracy_score(target_test, test_preds)



print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))

print('Validation Accuracy: {:.4}%'.format(test_accuracy * 100))
# prepare submission test data

column_diff = np.setdiff1d(test.columns.values, train_clean_standarized.columns.values)

test_clean = test.drop(column_diff, axis=1)



# XGBoost predictions

pred = clf.predict(test_clean)

submission = pd.DataFrame({"ID":test_clean.index, "TARGET":pred})

#submission.to_csv("submission_DecisionTree.csv", index=False)

submission.TARGET.value_counts(0)
fea_imp = pd.DataFrame({'imp': clf.feature_importances_, 'col': X.columns})

fea_imp = fea_imp[fea_imp.imp > .02].sort_values(['imp', 'col'], ascending=[True, False])

fea_imp.plot(kind='barh', x='col', y='imp', legend=None)

plt.title('XGBoost Tree - Feature importance')

plt.ylabel('Features')

plt.xlabel('Importance');
param_grid = {

    'max_depth': [2], #[2,3,4]

    'subsample': [0.6], #[0.4,0.5,0.6,0.7],

    'colsample_bytree': [0.5], #[0.5,0.6],

    'n_estimators': [100], #[100,200]

    'reg_alpha': [0.03] #[0.01, 0.02, 0.03, 0.04]

}



xgb_clf = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring="f1_weighted")

xgb_clf.fit(data_train, target_train)

best_est = xgb_clf.best_estimator_

print(best_est)
# predict on training and test sets

training_preds = xgb_clf.predict(data_train)

test_preds = xgb_clf.predict(data_test)



# accuracy of training and test sets

training_accuracy = accuracy_score(target_train, training_preds)

test_accuracy = accuracy_score(target_test, test_preds)



print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))

print('Validation Accuracy: {:.4}%'.format(test_accuracy * 100))
# prepare submission test data

column_diff = np.setdiff1d(test.columns.values, train_clean_standarized.columns.values)

test_clean = test.drop(column_diff, axis=1)



# XGBoost predictions

pred = xgb_clf.predict(test_clean)

submission = pd.DataFrame({"ID":test_clean.index, "TARGET":pred})

submission.to_csv("submission_XGBoost.csv", index=False)

submission.TARGET.value_counts(0)