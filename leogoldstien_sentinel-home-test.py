# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # pretty plotting
sns.set()
#import sklearn
from sklearn.preprocessing import minmax_scale, LabelEncoder
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_score, precision_recall_curve, confusion_matrix, average_precision_score
from sklearn.pipeline import make_pipeline

import itertools


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
train_df.info()
train_df.head()
train_df.malicious.sum()
print(f'malicious proportion: {train_df.malicious.sum() / train_df.malicious.count():.0%}')
categorical_variables = ['f2', 'f3', 'f4']
unique_categorical_values = {}
for cat in categorical_variables:
    unique_categorical_values[cat] = train_df[cat].unique()
    print(unique_categorical_values[cat])
for cat in categorical_variables:
    train_df[cat] = train_df[cat].astype('category')
train_df[categorical_variables].info()
categorical_encoders = {}
for cat in categorical_variables:
    le = LabelEncoder()
    le.fit(train_df[cat].values)
    categorical_encoders[cat] = le
    train_df[cat] = pd.DataFrame(le.transform(train_df[cat].values))
train_df[categorical_variables].head(5)
quantitative_df = train_df.drop(['index', 'malicious'] + categorical_variables, axis=1)
quantitative_variables = quantitative_df.columns
sns.heatmap(quantitative_df.corr(), cmap='YlOrBr_r')
quantitative_df.describe()
zero_vars = []
for var in quantitative_variables:
    #print(f"var: {var}, max: {quantitative_df[var].max()}, min: {quantitative_df[var].min()}")
    if quantitative_df[var].max() == quantitative_df[var].min():
        zero_vars.append(var)
print(zero_vars)
quantitative_no_zero_df = quantitative_df.drop(zero_vars, axis=1)
quantitative_no_zero_df.describe()
unscaled_vars = ['f1', 'f5', 'f6', 'f8', 'f10', 'f16', 'f23', 'f24', 'f32', 'f33']
for var in unscaled_vars:
    quantitative_no_zero_df[var] = pd.DataFrame(minmax_scale(quantitative_no_zero_df[var].values));
quantitative_no_zero_df.describe()
corr = quantitative_no_zero_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(10,10));
sns.heatmap(corr, mask = mask, cmap='BrBG', square=True, linewidths=.5);
plt.yticks(rotation=0);
correlated_vars = ['f22', 'f26', 'f28', 'f38', 'f39', 'f40', 'f41', 'f27', ]
sns.pairplot(quantitative_no_zero_df[correlated_vars])
clean_train_df = train_df[['index', 'malicious'] + categorical_variables]
clean_train_df = pd.concat([clean_train_df, quantitative_no_zero_df], axis=1)
clean_train_df.head()
def plot_roc_curves(fpr_tpr_list):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    for p in fpr_tpr_list:
        plt.plot(p[0], p[1], label=p[2])
    #plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    for p in fpr_tpr_list:
        plt.plot(p[0], p[1], label=p[2])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()
X = clean_train_df.drop(['malicious'], axis=1).values
y = clean_train_df['malicious'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid overfitting
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)

n_estimator = 10

# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator, random_state=42)
rt_lr = LogisticRegression()
pipeline_rt_lr = make_pipeline(rt, rt_lr)
pipeline_rt_lr.fit(X_train, y_train)

y_pred_rt = pipeline_rt_lr.predict_proba(X_test)[:, 1]
fpr_rt_lr, tpr_rt_lr, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf_clf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder(categories='auto')
rf_lr = LogisticRegression()
rf_clf.fit(X_train, y_train)
rf_enc.fit(rf_clf.apply(X_train))
rf_lr.fit(rf_enc.transform(rf_clf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lr = rf_lr.predict_proba(rf_enc.transform(rf_clf.apply(X_test)))[:, 1]
fpr_rf_lr, tpr_rf_lr, _ = roc_curve(y_test, y_pred_rf_lr)

# Gradient Boosting
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder(categories='auto')
grd_lr = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lr.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lr.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lr, tpr_grd_lr, _ = roc_curve(y_test, y_pred_grd_lm)

# The gradient boosted model by itself
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)

# The random forest model by itself
y_pred_rf = rf_clf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

# Naive Bayes classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
y_pred_nb = nb_clf.predict_proba(X_test)[:, 1]
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_pred_nb)

# Voting classifier
voting_clf = VotingClassifier(voting='soft', estimators=[('RF', RandomForestClassifier(max_depth=3, n_estimators=n_estimator)),
                                                         ('GBT', GradientBoostingClassifier(n_estimators=n_estimator)),
                                                         ('LR', LogisticRegression())])
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict_proba(X_test)[:, 1]
fpr_voting, tpr_voting, _ = roc_curve(y_test, y_pred_voting)
roc_data = [[fpr_rt_lr, tpr_rt_lr, 'RT + LR'], 
            [fpr_rf, tpr_rf, 'RF'],
            [fpr_rf_lr, tpr_rf_lr, 'RF + LR'],
            [fpr_grd, tpr_grd, 'GBT'],
            [fpr_grd_lr, tpr_grd_lr, 'GBT + LR'],
            [fpr_nb, tpr_nb, 'NB'],
            [fpr_voting, tpr_voting, 'Voting']]

plot_roc_curves(roc_data)
roc_data_voting = [[fpr_rt_lr, tpr_rt_lr, 'RT + LR'], 
                   [fpr_grd_lr, tpr_grd_lr, 'GBT + LR'],
                   [fpr_voting, tpr_voting, 'Voting']]
plot_roc_curves(roc_data_voting)
class GBLR_classifier():
    def __init__(self, n_estimators=10):
        self.grd = GradientBoostingClassifier(n_estimators=n_estimator)
        self.grd_enc = OneHotEncoder(categories='auto')
        self.grd_lr = LogisticRegression()
        
    def fit(self, X_train, y_train):
        # It is important to train the ensemble of trees on a different subset
        # of the training data than the linear regression model to avoid overfitting
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=0.5)
        self.grd.fit(X_train, y_train)
        self.grd_enc.fit(self.grd.apply(X_train)[:, :, 0])
        self.grd_lr.fit(self.grd_enc.transform(self.grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
        
    def predict_proba(self, X):
        return self.grd_lr.predict_proba(self.grd_enc.transform(self.grd.apply(X)[:, :, 0]))
    
    def predict(self, X):
        return self.grd_lr.predict(self.grd_enc.transform(self.grd.apply(X)[:, :, 0]))
clf = GBLR_classifier(n_estimators=10)
X = clean_train_df.drop(['malicious'], axis=1).values
y = clean_train_df['malicious'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf.fit(X_train, y_train)
clf_proba_predictions = clf.predict_proba(X_test)
clf_max_proba = np.amax(clf_proba_predictions, axis=1)
y_pred = clf.predict(X_test)
f1_score(y_pred=y_pred, y_true=y_test)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plot_confusion_matrix(confusion_matrix(y_pred=y_pred, y_true=y_test),
                      classes=['Safe', 'Mal'],
                      normalize=True)
precision, recall, thresholds = precision_recall_curve(y_test, clf_max_proba)
average_precision = average_precision_score(y_test, clf_max_proba)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: Average Precision={0:0.2f}'.format(average_precision));
test_df = pd.read_csv('../input/test.csv')
test_df.head()
for cat in categorical_variables:
    print(cat, [i for i in test_df[cat].unique() if i not in unique_categorical_values[cat]])
test_df = test_df.loc[~test_df['f3'].isin(['csnet_ns', 'link', 'supdup', 'sunrpc'])]
for cat in categorical_variables:
    test_df[cat] = pd.DataFrame(categorical_encoders[cat].transform(test_df[cat].values))
test_df.head()
test_df = test_df.drop(zero_vars, axis=1)
test_df.describe()
unscaled_vars = ['f1', 'f5', 'f6', 'f8', 'f10', 'f16', 'f23', 'f24', 'f32', 'f33']
for var in unscaled_vars:
    test_df[var] = pd.DataFrame(minmax_scale(test_df[var].values));
test_df.describe()
test_df.dropna(inplace=True)
test_pred = clf.predict(test_df.values)
submission_df = pd.DataFrame(test_pred, index=test_df['index'].values)
submission_df = submission_df.reset_index()
submission_df.columns = ['index', 'malicious']
submission_df.head()
submission_df.to_csv('submission.csv', index=False)
print(os.listdir("../working"))
