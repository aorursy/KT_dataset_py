# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Function from LAB  in CS4055



# Input



# folds - number of folds, i.e. the value of k, for k-fold cross-validation

# p - predictor attributes

# t - target attribute for 2-class classification

# classifier - a binary probabilistic classifier;

#              it is assumed that the there are two classes: 0 and 1

#              and the classifier learns to predict probabilities for the examples to belong to class 0 

#              as well as probabilities for the examples to belong to class 1

# mean_fpr - an array of equally spaced fpr values to be used for interpolating the tpr values



# Output



# _accuracies - average accuracy for each cross-validation run

# _f1_scores - F1 score for each cross-validation run 

# _tprs - a matrix of true positive rates, each row corresponds to a cross-validation run 

#         and contains 100 values, corresponding to equally spaced false positive rates in the array mean_fpr

# _aucs - areas under the curve, one per cross-validation run



def evaluate_classifier(folds, p, t, classifier, mean_fpr):

    _accuracies = np.array([])

    _f1_scores = np.array([])

    _tprs = np.empty(shape=[0,mean_fpr.shape[0]])

    _aucs = np.array([])

    

    # cv is a k-fold cross-valiatidation object

    cv = StratifiedKFold(n_splits=folds)

        

    for train_index, test_index in cv.split(p, t):

        

        # scale all predictor values to the range [0, 1] separately for the training and the test folds

        # note the target attribute 'type' is already binary        

        p_train = min_max_scaler.fit_transform(p[train_index,:])

        p_test = min_max_scaler.fit_transform(p[test_index,:])

        

        # train the classifier and compute the classes for the test set

        _model = classifier.fit(p_train, t[train_index])

        _probabilities = _model.predict_proba(p_test)

        _predictions = _model.predict(p_test)

        

        # compute accuracy

        _accuracies = np.append(_accuracies, accuracy_score(t[test_index], _predictions))

        

        # compute f1 score

        _f1_scores = np.append(_f1_scores, f1_score(t[test_index], _predictions))

    

        # compute fpr and tpr values for various thresholds 

        # by comparing the true target values to the predicted probabilities for class 1

        _fpr, _tpr, _thresholds = roc_curve(y_true = t[test_index], y_score = _probabilities[:, 1])

                        

        # compute true positive rates for the values in the array mean_fpr

        _tpr_transformed = np.array([interp(mean_fpr, _fpr, _tpr)])

        _tprs = np.concatenate((_tprs, _tpr_transformed), axis=0)

    

        # compute the area under the curve

        _aucs = np.append(_aucs, auc(_fpr, _tpr))

        

    return _accuracies, _f1_scores, _tprs, _aucs
def plot_roc_cv_folds(mean_fpr, tprs, aucs, classifier_name):

    

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    

    for i in range(0, aucs.shape[0]):

        plt.plot(mean_fpr, tprs[i,:], lw=1, alpha=0.3,label='fold %d (AUC = %0.2f)' % (i, aucs[i]))

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC curves for %d cross-validation folds: %s' % (aucs.shape[0], classifier_name))

    plt.legend(loc="lower right")

    plt.show()
def plot_roc_mean(mean_fpr, tprs, aucs, classifier_name):

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)



    mean_tpr = np.mean(tprs, axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)



    # ROC curve - mean curve for all cross-validation runs

    plt.plot(mean_fpr, mean_tpr, color='b',

             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

             lw=2, alpha=.8)



    # colour in grey the area of the standard deviation from the mean tpr

    std_tpr = np.std(tprs, axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')



    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Mean ROC curve for all cross-validation runs: ' + classifier_name)

    plt.legend(loc="lower right")

    plt.show()
def plot_roc_multiple_classifiers(mean_fpr, tprs, aucs, classifier_names):

    

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    

    for i in range(0, aucs.shape[0]):

        plt.plot(mean_fpr, tprs[i,:], lw=2, alpha=0.8,label='%s (AUC = %0.2f)' % (classifier_names[i], aucs[i]))

    

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC curves for multiple classifiers')

    plt.legend(loc="lower right")

    plt.show()