import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import *
from copy import deepcopy
train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")
print("Training set size: (%d, %d)" %(train.shape[0], train.shape[1]))
print("Test set size: (%d, %d)" %(test.shape[0], test.shape[1]))
# Visualisation options
pd.set_option('display.max_columns', None)
train.head()
summary = pd.DataFrame()
summary['Name'] = train.columns
summary['Type'] = train.dtypes.values
summary['Missing'] = train.isna().sum().values    
summary['Uniques'] = train.nunique().values
print(summary.to_string(index=False))
full_summary = pd.DataFrame(columns=["Train categories", "Test categories", "Not in test", "Not in train"],
                            index=train.drop(columns=['id','target']).columns)
for i, column in enumerate(train.drop(columns=['id','target'])):
    full_summary.iloc[i, 0] = len(train[column].value_counts())
    full_summary.iloc[i, 1] = len(test[column].value_counts())
    train_categories = set(train[column].value_counts().keys())
    test_categories = set(test[column].value_counts().keys())
    full_summary.iloc[i, 2] = len(train_categories - test_categories)
    full_summary.iloc[i, 3] = len(test_categories - train_categories)

print(full_summary)
# Get target labels and test id
train_labels = train.target
test_id = test.id

# ID column and target not necessary
train = train.drop(columns=['id', 'target'])
test = test.drop(columns=['id'])
fullset = pd.concat([train, test])
for column in fullset.columns:
    fullset[column] = fullset[column].fillna("NULL").astype(str)
ohe_encoder = preprocessing.OneHotEncoder(dtype=np.int8).fit(fullset)
fullset = ohe_encoder.transform(fullset)
train = fullset[:train.shape[0]]
test = fullset[train.shape[0]:]
var_filter = feature_selection.VarianceThreshold(threshold=0).fit(train)
train = var_filter.transform(train)
test = var_filter.transform(test)
# Univariate features filter
univariate_selection = feature_selection.SelectKBest(k='all').fit(train, train_labels)
plt.plot(np.cumsum(sorted(univariate_selection.scores_ / sum(univariate_selection.scores_), reverse=True)))
plt.xlabel("Number of features")
plt.ylabel("% of k-score obtained")
plt.title("Number of features (sorted by k-score) vs % of k-score explained")
plt.show()
kscore_filter = feature_selection.SelectKBest(k=1900).fit(train, train_labels)
train = kscore_filter.transform(train)
test = kscore_filter.transform(test)
splitter = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.05)
classifier = linear_model.LogisticRegression(max_iter=1e+5)
tuning_grid = {"C": (0.01, 0.1)}
grid_searcher = model_selection.GridSearchCV(estimator=classifier, param_grid=tuning_grid,
                                             scoring='roc_auc', cv=splitter, return_train_score=True)
model = grid_searcher.fit(train, train_labels)
print("Best hyperparameter: %s" %model.best_params_)
print("Best score: %.4f" %model.best_score_)
model = linear_model.LogisticRegression(max_iter=1e+6, C=0.1).fit(train, train_labels)
predictions = model.predict_proba(test)[:, 1]
submission = pd.DataFrame({'id': test_id, 'target': predictions})
submission.to_csv('submission.csv', index=False)