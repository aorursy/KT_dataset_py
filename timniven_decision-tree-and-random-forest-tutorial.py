import pandas as pd
import numpy as np
import random
from sklearn import model_selection, metrics, tree, ensemble
import matplotlib.pyplot as plt
file_name = '../input/SAheart.data'
data = pd.read_csv(file_name, index_col=0)
data.head()
data['famhist'] = data['famhist'] == 'Present'
random.seed(42)
np.random.seed(42)
train, test = model_selection.train_test_split(data)
x_train = train.loc[:, train.columns != 'chd']
y_train = train['chd']
x_test = test.loc[:, test.columns != 'chd']
y_test = test['chd']
splits = (x_train, y_train, x_test, y_test)
def train_and_evaluate(model, splits):
    x_train, y_train, x_test, y_test = splits
    model.fit(x_train, y_train)
    preds_train = model_bl.predict(x_train)
    preds_test = model_bl.predict(x_test)
    acc_train = metrics.accuracy_score(y_train, preds_train)
    acc_test = metrics.accuracy_score(y_test, preds_test)
    print('Training accuracy: %s' % acc_train)
    print('Testing accuracy: %s' % acc_test)

model_bl = tree.DecisionTreeClassifier(class_weight='balanced')
train_and_evaluate(model_bl, splits)
model_bl = tree.DecisionTreeClassifier(max_depth=5, class_weight='balanced')
train_and_evaluate(model_bl, splits)
model_bl = tree.DecisionTreeClassifier(max_depth=1, class_weight='balanced')
train_and_evaluate(model_bl, splits)
model_bl = tree.DecisionTreeClassifier(max_depth=3, class_weight='balanced')
train_and_evaluate(model_bl, splits)
def decisions(tree):
    n = len(tree.feature)
    names = x_train.columns
    if n < 8:
        raise ValueError('Expected tree of at least depth 3')
    print('            %s >= %s            ' % (names[tree.feature[0]], tree.threshold[0]))
    print('        /              \\       ')
    print('     %s >= %s         %s >= %s  ' % (names[tree.feature[1]], tree.threshold[1],
                                                names[tree.feature[2]], tree.threshold[2]))
    print('    /         \\     /         \\')
    print('%s >= %s  %s >= %s  %s >= %s  %s >= %s' % (names[tree.feature[3]], tree.threshold[3],
                                                      names[tree.feature[4]], tree.threshold[4],
                                                      names[tree.feature[5]], tree.threshold[5],
                                                      names[tree.feature[6]], tree.threshold[6]))
decisions(model_bl.tree_)
"""
Task: Experiment with the decision tree hyperparameters
Things to try:
  - min_samples_split
  - min_samples_leaf
  - max_features
  - max_leaf_nodes
"""
model_bl = tree.DecisionTreeClassifier(class_weight='balanced',
    max_depth=3)  # experiment with the parameters here
train_and_evaluate(model_bl, splits)
# this cell may take a minute to load
def scorer(model, X, y):
    preds = model.predict(X)
    return metrics.accuracy_score(y, preds)

max_n = 50
accs = []
for n in range(1, max_n):
    model = ensemble.RandomForestClassifier(
        max_depth=3, n_estimators=n, class_weight='balanced')
    acc = model_selection.cross_val_score(model, x_train, y_train, cv=10, scoring=scorer)
    accs.append(acc.mean())

plt.plot(range(1, max_n), accs)
plt.xlabel('n_estimators')
plt.ylabel('cross validated accuracy')
plt.show()
"""
Task: 
  Find your best random forest model. 
  Beat my baseline of 68.10. 
  How does your performance compare to logistic regression?
"""
def get_feature_set(data, wanted_features):
    return data.loc[:, [col in wanted_features for col in data.columns]]

def split(data):
    # control randomization for reproducibility
    np.random.seed(42)
    random.seed(42)
    train, test = model_selection.train_test_split(data)
    x_train = train.loc[:, train.columns != 'chd']
    y_train = train['chd']
    x_test = test.loc[:, test.columns != 'chd']
    y_test = test['chd']
    return x_train, y_train, x_test, y_test

def best_grid_model(param_grid, x_train, y_train):
    grid_search = model_selection.GridSearchCV(       # perform grid search
        estimator=ensemble.RandomForestClassifier(class_weight='balanced'),
        param_grid=param_grid,
        cv=10,
        return_train_score=True)
    grid_search.fit(x_train, y_train)
    best_model = ensemble.RandomForestClassifier(
        class_weight='balanced', **grid_search.best_params_)
    best_model.fit(x_train, y_train)
    print(grid_search.best_params_)
    return best_model

def evaluate(model, x_train, y_train, x_test, y_test):
    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)
    train_acc = metrics.accuracy_score(y_train, train_preds)
    test_acc = metrics.accuracy_score(y_test, test_preds)
    print('Train accuracy: %s' % train_acc)
    print('Test accuracy: %s' % test_acc)
wanted_features = data.columns
my_data = get_feature_set(data, wanted_features)  # feature selection
x_train, y_train, x_test, y_test = split(my_data) # splits
# update this!
param_grid = {
    'max_depth': [None],
    'n_estimators': [10]
}

best_model = best_grid_model(param_grid, x_train, y_train)
evaluate(best_model, x_train, y_train, x_test, y_test)