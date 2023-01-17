import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/pmr-3508-tarefa-2"))

# Any results you write to the current directory are saved as output.
print(os.listdir(".."))
dataset = pd.read_csv('../input/pmr-3508-tarefa-2/train_data.csv')
dataset.head()
sub_set_0 = dataset[['Id', 'ham']]
sub_set_0['ham'].value_counts().plot(kind='bar')
plt.xlabel('Is ham?')
plt.ylabel('Frequency')
plt.xticks(rotation='horizontal')
plt.tight_layout()
plt.show()
info = sub_set_0['ham'].value_counts()
print('{0:.2f}% of data correspond to spam emails'.format( info[0]*100/np.sum(info) ))
print('{0:.2f}% of data correspond to non spam emails'.format( info[1]*100/np.sum(info) ))
# Ref: 
# https://datascience.stackexchange.com/questions/10459/calculation-and-visualization-of-correlation-matrix-with-pandas
# https://matplotlib.org/examples/color/colormaps_reference.html

def correlation_matrix(df):
 
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('seismic', 30)

    label = df.columns[0]
    series = df[label]
    series = series.apply(lambda x: -x)
  
    df = pd.concat([df, series], axis=1)

    data = df.corr()
    cax = ax1.imshow(data, interpolation = "nearest", cmap=cmap)
    # ax1.grid(True)
    plt.title('Correlation matix')
    labels = df.columns
    # ax1.set_xticklabels(labels,fontsize=8, rotation=-45)
    # ax1.set_yticklabels(labels,fontsize=8)
    
    fig.colorbar(cax)
    plt.tight_layout()
    plt.show()

labels = dataset.columns
correlation_matrix(dataset[labels[:-2]])
THRESHOLD = 0.6
labels = dataset.columns
sub_set_1 = dataset[labels]

for label in labels:

    correlations = sub_set_1.corr()
    correlations.head()

    series = correlations[label]
    if abs(series.sort_values(ascending=False)[1]) > THRESHOLD:
        sub_set_1.drop(label, axis=1, inplace=True)

correlation_matrix(sub_set_1)
new_labels = sub_set_1.columns
print(len(new_labels))
print(len(labels))
def overall_performace(labels):
    
    gnb = GaussianNB()
    sig_labels = labels[:-2]

    scores = cross_val_score(gnb,
                            dataset[sig_labels], 
                            dataset['ham'], 
                            cv=10, 
                            n_jobs=-1,
                            scoring='roc_auc')
    print('Gaussian: {0:.2f}%'.format(np.mean(scores)*100))

    gnb = BernoulliNB()

    scores = cross_val_score(gnb,
                            dataset[sig_labels], 
                            dataset['ham'], 
                            cv=10, 
                            n_jobs=-1,
                            scoring='roc_auc')
    print('Bernoulli: {0:.2f}%'.format(np.mean(scores)*100))

    gnb = MultinomialNB()

    scores = cross_val_score(gnb,
                            dataset[sig_labels], 
                            dataset['ham'], 
                            cv=10, 
                            n_jobs=-1,
                            scoring='roc_auc')
    print('Multinomial: {0:.2f}%'.format(np.mean(scores)*100))

overall_performace(new_labels)
overall_performace(labels)
# Ref: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
bnl = BernoulliNB()
# bnl = GaussianNB()

X = dataset[sig_labels]
y = dataset['ham']

def cross_roc_curve(bnl, X, y):
    
    n_samples, n_features = X.shape
    cv = StratifiedKFold(n_splits=6)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):

        bnl.fit(X.iloc[train], y.iloc[train])
        probas_ = bnl.predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.axes().set_aspect('equal', 'datalim')
    plt.show()

cross_roc_curve(bnl, X, y)
X = dataset[labels[:-2]]
bnl = BernoulliNB()

cross_roc_curve(bnl, X, y)
test = pd.read_csv('../input/pmr-3508-tarefa-2/test_features.csv')
test.head()
bnl = BernoulliNB()

X = dataset[labels[:-2]]
y = dataset['ham']
bnl.fit(X, y)

X_test = test[labels[:-2]]
y_test = bnl.predict(X_test)

output = pd.concat( [test['Id'], pd.Series(y_test, name='ham')], axis=1)
output.to_csv('./summit1.csv', index=False)
bnl = BernoulliNB()

X = dataset[new_labels[:-2]]
y = dataset['ham']
bnl.fit(X, y)

X_test = test[new_labels[:-2]]
y_test = bnl.predict(X_test)

output2 = pd.concat( [test['Id'], pd.Series(y_test, name='ham')], axis=1)
output2.to_csv('./summit2.csv', index=False)
bnl = GausssianNB()

X = dataset[new_labels[:-2]]
y = dataset['ham']
bnl.fit(X, y)

X_test = test[new_labels[:-2]]
y_test = bnl.predict(X_test)

output2 = pd.concat( [test['Id'], pd.Series(y_test, name='ham')], axis=1)
output2.to_csv('./summit3.csv', index=False)
THRESHOLD = 0.5
labels = dataset.columns
sub_set_2 = dataset[labels]

for label in labels:

    correlations = sub_set_2.corr()
    correlations.head()

    series = correlations[label]
    if abs(series.sort_values(ascending=False)[1]) > THRESHOLD:
        sub_set_2.drop(label, axis=1, inplace=True)

new_labels_2 = sub_set_2.columns
correlation_matrix(sub_set_2)
overall_performace(new_labels_2)
