# standard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#To create more features
from itertools import combinations, product 

#Needed for the tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

#Needed for the forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

#Needed for XGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve

import warnings
warnings.filterwarnings('ignore') #because we are animals
df = pd.read_csv('../input/UCI_Credit_Card.csv')
df.columns
df['default.payment.next.month'].value_counts() #77.88%
# This is just to speed up the training
del df['ID']
del df['BILL_AMT5']
del df['BILL_AMT6']
del df['PAY_AMT5']
del df['PAY_AMT6']
del df['PAY_5']
del df['PAY_6']
df.info()
# Create a bunch of new features
target = df['default.payment.next.month'].copy()
cols = df.columns[:-1]
data = df[cols].copy()

cc = list(combinations(data.columns,2))
datacomb = pd.concat([data[c[0]] * data[c[1]] for c in cc], axis=1, keys=cc)
datacomb.columns = datacomb.columns.map(''.join)

df_large = pd.concat([data, datacomb, target], axis = 1)
df_large.head()
y = df['default.payment.next.month'].copy()
X = df.copy().drop('default.payment.next.month', axis = 1)
XL = df_large.copy().drop('default.payment.next.month', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=895)
XL_train, XL_test, yL_train, yL_test = train_test_split(XL, y, test_size=0.33, random_state=895)
tree = DecisionTreeClassifier()
tree
tree.fit(X_train, y_train)
prediction = tree.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))
tree.fit(XL_train, yL_train)
prediction = tree.predict(XL_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))
dp_list = np.arange(3, 30)
train = []
test = []
trainL = []
testL = []

for depth in dp_list:
    tree = DecisionTreeClassifier(max_depth=depth)
    tree.fit(X_train, y_train)
    prediction = tree.predict(X_test)
    trainpred = tree.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    tree.fit(XL_train, yL_train)
    prediction = tree.predict(XL_test)
    trainpred = tree.predict(XL_train)
    train_acc = accuracy_score(yL_train, trainpred)
    test_acc = accuracy_score(yL_test, prediction)
    trainL.append(train_acc)
    testL.append(test_acc)
    
performance = pd.DataFrame({'max_depth':dp_list,'Train_acc':train,'Test_acc':test})
performanceL= pd.DataFrame({'max_depth':dp_list,'Train_acc':trainL,'Test_acc':testL})

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))
x_axis = dp_list
ax1.plot(x_axis, performance['Train_acc'], label='Train')
ax1.plot(x_axis, performance['Test_acc'], label='Test')
ax1.legend()
plt.ylabel('accuracy')
plt.title('Tree accuracy vs depth')
ax2.plot(x_axis, performanceL['Train_acc'], label='Train')
ax2.plot(x_axis, performanceL['Test_acc'], label='Test')
ax2.legend()
plt.ylabel('accuracy')
plt.title('Tree accuracy vs depth')
plt.show()
sam_list = np.arange(1,30)
train = []
test = []
trainL = []
testL = []

for sam in sam_list:
    tree = DecisionTreeClassifier(min_samples_leaf=sam)
    tree.fit(X_train, y_train)
    prediction = tree.predict(X_test)
    trainpred = tree.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    tree.fit(XL_train, yL_train)
    prediction = tree.predict(XL_test)
    trainpred = tree.predict(XL_train)
    train_acc = accuracy_score(yL_train, trainpred)
    test_acc = accuracy_score(yL_test, prediction)
    trainL.append(train_acc)
    testL.append(test_acc)
    
performance = pd.DataFrame({'min_samples_leaf':sam_list,'Train_acc':train,'Test_acc':test})
performanceL= pd.DataFrame({'min_samples_leaf':sam_list,'Train_acc':trainL,'Test_acc':testL})

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))
x_axis = sam_list
ax1.plot(x_axis, performance['Train_acc'], label='Train')
ax1.plot(x_axis, performance['Test_acc'], label='Test')
ax1.legend()
plt.ylabel('Accuracy')
plt.title('Tree accuracy vs min_sample_leaf')
ax2.plot(x_axis, performanceL['Train_acc'], label='Train')
ax2.plot(x_axis, performanceL['Test_acc'], label='Test')
ax2.legend()
plt.ylabel('Accuracy')
plt.title('Tree accuracy vs min_sample_leaf')
plt.show()
sam_list = np.arange(2,40, 2)
train = []
test = []
trainL = []
testL = []

for sam in sam_list:
    tree = DecisionTreeClassifier(min_samples_split=sam)
    tree.fit(X_train, y_train)
    prediction = tree.predict(X_test)
    trainpred = tree.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    tree.fit(XL_train, yL_train)
    prediction = tree.predict(XL_test)
    trainpred = tree.predict(XL_train)
    train_acc = accuracy_score(yL_train, trainpred)
    test_acc = accuracy_score(yL_test, prediction)
    trainL.append(train_acc)
    testL.append(test_acc)
    
performance = pd.DataFrame({'min_samples_split':sam_list,'Train_acc':train,'Test_acc':test})
performanceL= pd.DataFrame({'min_samples_split':sam_list,'Train_acc':trainL,'Test_acc':testL})

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))
x_axis = sam_list
ax1.plot(x_axis, performance['Train_acc'], label='Train')
ax1.plot(x_axis, performance['Test_acc'], label='Test')
ax1.legend()
plt.ylabel('Accuracy')
plt.title('Tree accuracy vs min_sample_split')
ax2.plot(x_axis, performanceL['Train_acc'], label='Train')
ax2.plot(x_axis, performanceL['Test_acc'], label='Test')
ax2.legend()
plt.ylabel('Accuracy')
plt.title('Tree accuracy vs min_sample_split')
plt.show()
skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=268)
#Don't run during class, the result is in the next cell.

param_grid = {'max_depth': np.arange(2,10),
              'min_samples_split' : np.arange(2,20,2),
              'min_samples_leaf' : np.arange(1,21,2),
             'random_state': [42]}

#create a grid
grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'accuracy', n_jobs=-1, cv=skf)

#training
%time grid_tree.fit(X_train, y_train)

#let's see the best estimator
best_tree = grid_tree.best_estimator_
print(best_tree)
print("_"*40)
#with its score
print("Cross-validated best score {}%".format(round(grid_tree.best_score_ * 100,3)))
#score on test
predictions = best_tree.predict(X_test)
print("Test score: {}%".format(round(accuracy_score(y_true = y_test, y_pred = predictions) * 100,3)))
best_tree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=13, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')
cm = confusion_matrix(y_test, predictions)
cmap = plt.cm.Blues
classes = [0,1]
thresh = cm.max() / 2.
fmt = 'd'

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()
target_names = ['Not Default', 'Default']
print(classification_report(y_test, prediction, target_names=target_names))
forest = RandomForestClassifier(n_jobs = -1, random_state=42)
forest
forest.fit(X_train, y_train)
prediction = forest.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))
def get_feature_importance(clsf, ftrs):
    imp = clsf.feature_importances_.tolist()
    feat = ftrs
    result = pd.DataFrame({'feat':feat,'score':imp})
    result = result.sort_values(by=['score'],ascending=False)
    return result
get_feature_importance(forest, X.columns).head(10)
forest.fit(XL_train, yL_train)
prediction = forest.predict(XL_test)
print("Accuracy: {}%".format(round(accuracy_score(yL_test, prediction) * 100,3)))
get_feature_importance(forest, XL.columns).head(10)
get_feature_importance(forest, XL.columns).tail(10)
dp_list = np.arange(3, 30)
train = []
test = []

for depth in dp_list:
    forest = RandomForestClassifier(max_depth=depth, n_jobs = -1, random_state=42)
    forest.fit(X_train, y_train)
    prediction = forest.predict(X_test)
    trainpred = forest.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'n_estimators':dp_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = dp_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('Forest accuracy vs depth')
plt.show()
tree_list = np.arange(3, 80)
train = []
test = []

for tree in tree_list:
    forest = RandomForestClassifier(n_estimators=tree, n_jobs = -1, random_state=42)
    forest.fit(X_train, y_train)
    prediction = forest.predict(X_test)
    trainpred = forest.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'n_estimators':tree_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = tree_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('Forest accuracy vs n_estimators')
plt.show()
leaf_list = np.arange(1, 100)
train = []
test = []

for leaf in leaf_list:
    forest = RandomForestClassifier(min_samples_leaf=leaf, n_jobs = -1, random_state=42)
    forest.fit(X_train, y_train)
    prediction = forest.predict(X_test)
    trainpred = forest.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'min_samples_leaf':leaf_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = leaf_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('Forest accuracy vs min_samples_leaf')
plt.show()
leaf_list = np.arange(2, 80, 2)
train = []
test = []

for leaf in leaf_list:
    forest = RandomForestClassifier(min_samples_split=leaf, n_jobs = -1, random_state=42)
    forest.fit(X_train, y_train)
    prediction = forest.predict(X_test)
    trainpred = forest.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'min_samples_split':leaf_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = leaf_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('Forest accuracy vs min_samples_split')
plt.show()
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 30, num = 15)]
max_depth.append(None)
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(2, 50, num = 25)]
# Minimum number of samples to split an internal node
min_samples_split = [int(x) for x in np.linspace(2, 100, num = 50)]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf,
               'min_samples_split' : min_samples_split,
               'n_jobs' : [-1],
               'random_state' : [42]}
# Don't run during class, the result is in the next cell.

grid_forest = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, 
                               n_iter = 50, cv = skf, random_state=42, n_jobs = -1,
                                scoring = 'accuracy')

#training
%time grid_forest.fit(X_train, y_train)

#let's see the best estimator
best_forest = grid_forest.best_estimator_
print(best_forest)
print("_"*40)
#with its score
print("Cross-validated best score {}%".format(round(grid_forest.best_score_ * 100,3)))
#score on test
predictions = best_forest.predict(X_test)
print("Test score: {}%".format(round(accuracy_score(y_true = y_test, y_pred = predictions) * 100,3)))
best_forest2 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=15, min_samples_split=30,
            min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=-1,
            oob_score=False, random_state=42, verbose=0, warm_start=False)

%time best_forest2.fit(X_train, y_train)

predictions = best_forest2.predict(X_test)
print("Test score: {}%".format(round(accuracy_score(y_true = y_test, y_pred = predictions) * 100,3)))
cm = confusion_matrix(y_test, predictions)
cmap = plt.cm.Blues
classes = [0,1]
thresh = cm.max() / 2.
fmt = 'd'

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()
target_names = ['Not Default', 'Default']
print(classification_report(y_test, prediction, target_names=target_names))
xgb = XGBClassifier(n_jobs = -1)
xgb
xgb.fit(X_train, y_train)
prediction = xgb.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))
tree_list = np.arange(10, 1000, 10) # 500500 trees...
train = []
test = []

for tree in tree_list:
    xgb = XGBClassifier(n_estimators=tree, n_jobs = -1, random_state=42)
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    trainpred = xgb.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'n_estimators':tree_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = tree_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('XGB accuracy vs n_estimators')
plt.show()
learn_list = np.arange(0.01, 0.99, 0.01) # About 20000 trees
train = []
test = []

for tree in learn_list:
    xgb = XGBClassifier(n_estimators=200, learning_rate=tree, n_jobs = -1, random_state=42)
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    trainpred = xgb.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'learning_rate':learn_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = learn_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('XGB accuracy vs learning_rate')
plt.show()
tree_list = np.arange(2, 10) 
train = []
test = []

for tree in tree_list:
    xgb = XGBClassifier(max_depth=tree, n_jobs = -1, random_state=42)
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    trainpred = xgb.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'max_depth':tree_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = tree_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('XGB accuracy vs max_depth')
plt.show()
tree_list = np.arange(0.1, 0.9, 0.1)
train = []
test = []
train2 = []
test2 = []

for tree in tree_list:
    xgb = XGBClassifier(base_score=tree, n_jobs = -1, random_state=42)
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    trainpred = xgb.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    xgb = XGBClassifier(base_score=tree, n_estimators=500, n_jobs = -1, random_state=42)
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    trainpred = xgb.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train2.append(train_acc)
    test2.append(test_acc)
    
performance = pd.DataFrame({'base_score':tree_list,'Train_acc':train,'Test_acc':test})
performance2 = pd.DataFrame({'base_score':tree_list,'Train_acc':train2,'Test_acc':test2})

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))

x_axis = tree_list
ax1.plot(x_axis, performance['Train_acc'], label='Train')
ax1.plot(x_axis, performance['Test_acc'], label='Test')
ax1.legend()
plt.ylabel('accuracy')
plt.title('Tree accuracy vs base_score')
ax2.plot(x_axis, performance2['Train_acc'], label='Train')
ax2.plot(x_axis, performance2['Test_acc'], label='Test')
ax2.legend()
plt.ylabel('accuracy')
plt.title('Tree accuracy vs base_score')
plt.show()
best_XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=-1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
best_XGB.fit(X_train, y_train)
prediction = best_XGB.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))
cm = confusion_matrix(y_test, predictions)
cmap = plt.cm.Blues
classes = [0,1]
thresh = cm.max() / 2.
fmt = 'd'

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()
target_names = ['Not Default', 'Default']
print(classification_report(y_test, prediction, target_names=target_names))
eval_set = [(X_train, y_train), (X_test, y_test)]

best_XGB.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=False)
# retrieve performance metrics
results = best_XGB.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot rmse
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('error')
plt.title('XGBoost accuracy')
plt.show()
eval_set = [(X_test, y_test)]
best_XGB.fit(X_train, y_train, early_stopping_rounds=200, 
             eval_metric="error", eval_set=eval_set, verbose=True)
best_XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=381,
       n_jobs=-1, nthread=None, objective='binary:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=1)
best_XGB.fit(X_train, y_train)
prediction = best_XGB.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))
cm = confusion_matrix(y_test, predictions)
cmap = plt.cm.Blues
classes = [0,1]
thresh = cm.max() / 2.
fmt = 'd'

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()
target_names = ['Not Default', 'Default']
print(classification_report(y_test, prediction, target_names=target_names))
train_sizes = [1, 500, 1000, 3000, 5000, 7000, 9000, 11000, 13000, 15599] #there are 19500 entries in the train set

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = best_tree, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70)
print('\nValidation scores:\n\n', validation_scores)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Decision tree', fontsize = 18, y = 1.03)
plt.legend()
train_sizes = [1, 500, 1000, 3000, 5000, 7000, 9000, 11000, 14399]

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = best_forest2, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70)
print('\nValidation scores:\n\n', validation_scores)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Random Forest', fontsize = 18, y = 1.03)
plt.legend()
train_sizes = [1, 500, 1000, 3000, 5000, 7000, 9000, 11000, 14399]

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = best_XGB, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70)
print('\nValidation scores:\n\n', validation_scores)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('XGBoost', fontsize = 18, y = 1.03)
plt.legend()
best_tree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=13, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')

#Change the next two

over_tree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=13, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')

under_tree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=13, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')
tree = best_tree #change here

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = tree, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Decision tree', fontsize = 18, y = 1.03)
plt.legend()

tree.fit(X_train, y_train)
prediction = tree.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))

print(classification_report(y_test, prediction, target_names=target_names))
best_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=15, min_samples_split=30,
            min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=-1,
            oob_score=False, random_state=42, verbose=0, warm_start=False)

# Change the next two

over_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=15, min_samples_split=30,
            min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=-1,
            oob_score=False, random_state=42, verbose=0, warm_start=False)

under_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=15, min_samples_split=30,
            min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=-1,
            oob_score=False, random_state=42, verbose=0, warm_start=False)
forest = best_forest #change here

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = forest, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Random Forest', fontsize = 18, y = 1.03)
plt.legend()

forest.fit(X_train, y_train)
prediction = forest.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))

print(classification_report(y_test, prediction, target_names=target_names))
best_XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=381,
       n_jobs=-1, nthread=None, objective='binary:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=1)

#change the next two

over_XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=381,
       n_jobs=-1, nthread=None, objective='binary:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=1)

under_XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=381,
       n_jobs=-1, nthread=None, objective='binary:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=1)
XGB = best_XGB #change here

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = XGB, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('XGBoost', fontsize = 18, y = 1.03)
plt.legend()

XGB.fit(X_train, y_train)
prediction = XGB.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))

print(classification_report(y_test, prediction, target_names=target_names))