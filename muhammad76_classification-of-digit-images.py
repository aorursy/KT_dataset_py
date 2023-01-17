#Hands on maching learning book : Chapter - 3 ( Project )
#Fatching data

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
#Watching number row and column
X, y = mnist["data"], mnist["target"]
X.shape
y.shape
#Getting a single digit image and print it
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()
#Watch the corresponding digit value of above image
y[0]
#Converting digit to its corresponding int value
import numpy as np
y = y.astype(np.uint8)
y[0]
#spliting test set and train set.
#The MNIST dataset is actually already split into a training set (the first 60,000 images)
# and a test set (the last 10,000 images):

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
#Testing is digit is 5 or not
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits
y_test_5 = (y_test == 5)
#SGD classifier and feeding it data
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
#predict is some_digit is 5 or not
sgd_clf.predict([some_digit])
#Cross validetion by dividing the data set in 3 part
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) # prints 0.9502, 0.96565, and 0.96495
#Using k-fold cross validation. Here we divide data in 3 fold and verify
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#Classifies all images in not 5 class

from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
#Cross validation using k-fold
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#Cross validation by confusion matrix
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
#Printing confusion matrix. 
#row is actual class
#column is predictied class
#1st row is non-5 (negetive) and actually non-5
#2nd row is 5 and actually 5
#1st colum is non-5 and classified as non-5, 2nd column is 5 and classified as 5
#53892 images are non-5 and classified as non-5 true negetive
#687 images are non-5 and classified as 5 false positive
#1891 images are 5 and classified as non-5 false negetive
#3530 images are 5 and classified as 5 true positve

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
#Let's check predending we get actuall answer perfectly, that means set the answer colum as question
y_train_perfect_predictions = y_train_5 # pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)
#precision = (TP / ( TP + FP))
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred) # == 3530 / (3530 + 687)
#recall = ( TP / (TP + FN ))
recall_score(y_train_5, y_train_pred) # == 3530 / (3530 + 1891)
#combine precision and recall into a single metric called the F1
#score, in particular if you need a simple way to compare two classifiers. The F1 score is
#the harmonic mean of precision and recall. Whereas the regular mean
#treats all values equally, the harmonic mean gives much more weight to low values.
#As a result, the classifier will only get a high F1 score if both recall and precision are high.
# F1 = 2 / ( 1 / precision ) + (1 / recall ) )
# = ( 2 × precision × recall ) / ( precision + recall )
# = TP / ( TP +  ( ( FN + FP ) / 2 ) )

from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)
#Get the digit score
y_scores = sgd_clf.decision_function([some_digit])
y_scores
#setting threshold score and checking the digit
threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
#setting threshold score and checking the digit
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
#Using decision_fucntion to get decision score and create curve
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

#Printing the curve
#precision blue
#recall green
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    [...] # highlight the threshold and add the legend, axis label, and grid
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
#Printing the curve for precision
#precision blue
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    [...] # highlight the threshold and add the legend, axis label, and grid
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
#Find lowest thresh hold that will give me 90% precision
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)] # ~3370
threshold_90_precision
#Prediction using 90% precision threshold
y_train_pred_90 = (y_scores >= threshold_90_precision)
#Precision score
precision_score(y_train_5, y_train_pred_90)
#Recall score
recall_score(y_train_5, y_train_pred_90)
#the FPR ( False positive rate ), X against the TPR ( True positive rate or recall), Y
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.show()
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
#Using random forest model
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,method="predict_proba")
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()
roc_auc_score(y_train_5, y_scores_forest)
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores_forest)
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3)
p = precision_score(y_train_5, y_train_pred_forest)
p
r = recall_score(y_train_5, y_train_pred_forest)
r
#Using Support vector machine
from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train) # y_train, not y_train_5
svm_clf.predict([some_digit])
#checking is 5 got the maximum score
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores
np.argmax(some_digit_scores)
svm_clf.classes_
svm_clf.classes_[5]
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
ovr_clf.predict([some_digit])
len(ovr_clf.estimators_)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
sgd_clf.decision_function([some_digit])
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)

plt.show()

from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)