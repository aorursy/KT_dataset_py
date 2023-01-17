import numpy as np

import pandas as pd



from sklearn.preprocessing import LabelBinarizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix



import matplotlib.pyplot as plt

plt.style.use("ggplot")



df = pd.read_csv('../input/data.csv')
# class distribution

# diagnosis: B = 0, M = 1

df['diagnosis'].value_counts()
# by default majority class (benign) will be negative

lb = LabelBinarizer()

df['diagnosis'] = lb.fit_transform(df['diagnosis'].values)

targets = df['diagnosis']



df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets)
print('y_train class distribution')

print(y_train.value_counts(normalize=True))



print('y_test class distribution')

print(y_test.value_counts(normalize=True))
clf = RandomForestClassifier(n_jobs=-1)



param_grid = {

    'min_samples_split': [3, 5, 10], 

    'n_estimators' : [100, 300],

    'max_depth': [3, 5, 15, 25],

    'max_features': [3, 5, 10, 20]

}
scorers = {

    'precision_score': make_scorer(precision_score),

    'recall_score': make_scorer(recall_score),

    'accuracy_score': make_scorer(accuracy_score)

}
def grid_search_wrapper(refit_score='precision_score'):

    """

    fits a GridSearchCV classifier using refit_score for optimization

    prints classifier performance metrics

    """

    skf = StratifiedKFold(n_splits=10)

    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,

                           cv=skf, return_train_score=True, n_jobs=-1)

    grid_search.fit(X_train.values, y_train.values)



    # make the predictions

    y_pred = grid_search.predict(X_test.values)



    print('Best params for {}'.format(refit_score))

    print(grid_search.best_params_)



    # confusion matrix on the test data.

    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))

    print(pd.DataFrame(confusion_matrix(y_test, y_pred),

                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

    return grid_search
grid_search_clf = grid_search_wrapper(refit_score='precision_score')
results = pd.DataFrame(grid_search_clf.cv_results_)

results = results.sort_values(by='mean_test_precision_score', ascending=False)

results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score',

         'param_max_depth', 'param_max_features', 'param_min_samples_split',

         'param_n_estimators']].head()
grid_search_clf = grid_search_wrapper(refit_score='recall_score')
results = pd.DataFrame(grid_search_clf.cv_results_)

results = results.sort_values(by='mean_test_recall_score', ascending=False)

results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score',

         'param_max_depth', 'param_max_features', 'param_min_samples_split',

         'param_n_estimators']].head()
# this gives the probability [0,1] that each sample belongs to class 1

y_scores = grid_search_clf.predict_proba(X_test)[:, 1]



# for classifiers with decision_function, this achieves similar results

# y_scores = classifier.decision_function(X_test)
def adjusted_classes(y_scores, t):

    """

    This function adjusts class predictions based on the prediction threshold (t).

    Will only work for binary classification problems.

    """

    return [1 if y >= t else 0 for y in y_scores]
# generate the precision recall curve

p, r, thresholds = precision_recall_curve(y_test, y_scores)
def precision_recall_threshold(t=0.5):

    """

    plots the precision recall curve and shows the current value for each

    by identifying the classifier's threshold (t).

    """

    

    # generate new class predictions based on the adjusted_classes

    # function above and view the resulting confusion matrix.

    y_pred_adj = adjusted_classes(y_scores, t)

    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),

                       columns=['pred_neg', 'pred_pos'], 

                       index=['neg', 'pos']))

    

    # plot the curve

    plt.figure(figsize=(8,8))

    plt.title("Precision and Recall curve ^ = current threshold")

    plt.step(r, p, color='b', alpha=0.2,

             where='post')

    plt.fill_between(r, p, step='post', alpha=0.2,

                     color='b')

    plt.ylim([0.5, 1.01]);

    plt.xlim([0.5, 1.01]);

    plt.xlabel('Recall');

    plt.ylabel('Precision');

    

    # plot the current threshold on the line

    close_default_clf = np.argmin(np.abs(thresholds - t))

    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',

            markersize=15)
# The best I could do with 1 FN was 0.17, but re-execute to watch the confusion matrix change.

precision_recall_threshold(0.17)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    """

    Modified from:

    Hands-On Machine learning with Scikit-Learn

    and TensorFlow; p.89

    """

    plt.figure(figsize=(8, 8))

    plt.title("Precision and Recall Scores as a function of the decision threshold")

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

    plt.ylabel("Score")

    plt.xlabel("Decision Threshold")

    plt.legend(loc='best')
# use the same p, r, thresholds that were previously calculated

plot_precision_recall_vs_threshold(p, r, thresholds)
def plot_roc_curve(fpr, tpr, label=None):

    """

    The ROC curve, modified from 

    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91

    """

    plt.figure(figsize=(8,8))

    plt.title('ROC Curve')

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.005, 1, 0, 1.005])

    plt.xticks(np.arange(0,1, 0.05), rotation=90)

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate (Recall)")

    plt.legend(loc='best')
fpr, tpr, auc_thresholds = roc_curve(y_test, y_scores)

print(auc(fpr, tpr)) # AUC of ROC

plot_roc_curve(fpr, tpr, 'recall_optimized')