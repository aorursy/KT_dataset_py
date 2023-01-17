import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print(train.shape)
print(test.shape)
# Visualize some pictures
eg_digit_image = train.iloc[10000, 1:].values.reshape(28, 28)
eg_digit_label = train.iloc[10000, 0]
plt.imshow(eg_digit_image, cmap = mpl.cm.binary, interpolation = 'nearest')
plt.axis('off')
plt.show()
print(eg_digit_label)
# shuffle train data
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train.iloc[:, 1:].values, train.iloc[:, 0].values, test_size = 0.2, random_state = 777)
shuffle_index = np.random.permutation(x_train.shape[0])
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == 5)
y_valid_5 = (y_valid == 5)
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(x_train, y_train_5)
some_digits = train[train['label'] == 5].drop(columns = 'label').values
sgd_clf.predict(some_digits)
# DIY cross validation
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits = 3, random_state = 42)
for train_index, test_index in skfolds.split(x_train, y_train_5):
    clone_clf = clone(sgd_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = y_train_5[train_index]
    x_test_fold = x_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
# use cross_val_score
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, x_train, y_train_5, cv = 3, scoring = 'accuracy')
# cross_val_predict, confusion_matrix, precision and recall
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv = 3)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: ', precision_score(y_train_5, y_train_pred))
print('Recall: ', recall_score(y_train_5, y_train_pred))
print('F1: ', f1_score(y_train_5, y_train_pred))
recall_score(y_train_5, y_train_pred)
y_scores = sgd_clf.decision_function([some_digits[0]])
y_scores
threshold = 200000
y_some_digits_pred = (y_scores > threshold)
y_some_digits_pred
y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv = 3, method = 'decision_function')
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label = 'Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label = 'Recall')
    plt.xlabel('Threshold')
    plt.legend(loc = 'center left')
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
precisions
plt.plot(recalls, precisions)
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.ylim([0, 1])
plt.xlim([0, 1])
# roc curve
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
# AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
# compare RF to SDG
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv = 3, method = 'predict_proba')
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, 'b:', label = 'SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc = 'lower right')
y_label_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv = 3)
print('AUC: ', roc_auc_score(y_train_5, y_scores_forest))
print('Precision: ', precision_score(y_train_5, y_label_forest))
print('Recall: ', recall_score(y_train_5, y_label_forest))
# use binary classifier for multiclass classification - sklearn automatically detects this
sgd_clf.fit(x_train, y_train)
sgd_clf.predict([some_digits[0]])
sgd_clf.decision_function([some_digits[0]])
sgd_clf.classes_
# force SGD to use OvO strategy
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state = 42))
ovo_clf.fit(x_train, y_train)
ovo_clf.predict([some_digits[0]])
len(ovo_clf.estimators_)
# random forest
forest_clf.fit(x_train, y_train)
forest_clf.predict([some_digits[0]])
forest_clf.predict_proba([some_digits[0]])
# evaluate the classifiers
cross_val_score(sgd_clf, x_train, y_train, cv = 3, scoring = 'accuracy')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
cross_val_score(sgd_clf, x_train_scaled, y_train, cv = 3, scoring = 'accuracy')
# error analysis
y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train, cv = 3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
plt.matshow(conf_mx, cmap = plt.cm.gray)
row_sums = conf_mx.sum(axis = 1, keepdims = True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap = plt.cm.gray)
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_multilabel)
knn_clf.predict([some_digits[0]])
y_train_knn_pred = cross_val_predict(knn_clf, x_train, y_multilabel, cv = 3)
f1_score(y_multilabel, y_train_knn_pred, average = 'macro')
f1_score(y_multilabel, y_train_knn_pred, average = 'weighted')
noise = np.random.randint(0, 100, (len(x_train), 784))
x_train_mod = x_train + noise
noise = np.random.randint(0, 100, (len(x_valid), 784))
x_valid_mod = x_valid + noise
y_train_mod = x_train
y_valid_mod = x_valid
x_train_mod[0].shape
plt.imshow(x_train_mod[0].reshape(28, 28), cmap = mpl.cm.binary, interpolation = 'nearest')
plt.axis('off')
plt.imshow(x_train[0].reshape(28, 28), cmap = mpl.cm.binary, interpolation = 'nearest')
plt.axis('off')
plt.imshow(x_valid[0].reshape(28, 28), cmap = mpl.cm.binary, interpolation = 'nearest')
plt.axis('off')
knn_clf.fit(x_train_mod, y_train_mod)
clean_digit = knn_clf.predict([x_valid_mod[0]])
plt.imshow(clean_digit.reshape(28, 28), cmap = mpl.cm.binary, interpolation = 'nearest')
plt.axis('off')
xTrain = train.iloc[:, 1:]
yTrain = train.iloc[:, 0]
xTrainScaled = scaler.fit_transform(xTrain)
xTestScaled = scaler.transform(test)
knn = KNeighborsClassifier(weights = 'distance', n_neighbors = 4)
knn.fit(xTrainScaled, yTrain)
testPred = knn.predict(xTestScaled)
result = pd.concat([pd.DataFrame(np.arange(0, len(test)) + 1), pd.DataFrame(testPred)], axis = 1)
result.columns = ['ImageId', 'Label']
result.to_csv('digit_recognizer_20200705.csv', index = False)
result