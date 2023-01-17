# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import itertools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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
def plot_roc_curve(X_test_df_result):
    fpr, tpr, threshold = metrics.roc_curve(X_test_df_result['Actuals'], X_test_df_result['TARGET'])
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return plt

df = pd.read_csv("../input/train.csv")
unseen_df = pd.read_csv('../input/test.csv')
df_submission = pd.read_csv('../input/sample_submission.csv')

df.head(3)
df_X = df.drop(['TARGET'], axis=1)
df_Y = df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.30, random_state=2018)
print("Size of features in training and test set is %s | %s" % (X_train.shape, X_test.shape))
print("Size of labeled data in training and test set is %s | %s" % (y_train.shape, y_test.shape))
y_test[y_test == 0].shape
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, min_samples_leaf=1, oob_score=True)
clf = clf.fit(X_train, y_train)
print("Validation set score - %f " % clf.score(X_test, y_test))
print("Out of bag score - %f" % clf.oob_score_)

def predict_and_prepare_df_to_submit(unseen_df):
    result = clf.predict(unseen_df)
    ids = unseen_df['ID'].copy()
    submit_df = pd.DataFrame({'ID': ids, 'TARGET': result }, columns=['ID', 'TARGET'])
    return submit_df

df_to_submit = predict_and_prepare_df_to_submit(unseen_df)
df_to_submit[df_to_submit['TARGET'] == 1]

X_test_df_result = predict_and_prepare_df_to_submit(X_test)
# Add actual value for comparision
X_test_df_result['Actuals'] = y_test;
(positive_entries, _) = X_test_df_result[X_test_df_result['Actuals'] == 1].shape
(total_entries, _) = X_test_df_result.shape
print('%f%% of %d entries are positive.' % ( ((positive_entries * 100) / total_entries), total_entries ))
y_test.shape[0]
plot_roc_curve(X_test_df_result).show()
# print('AuROC is %f' % metrics.roc_auc_score(X_test_df_result['Actuals'], X_test_df_result['TARGET']));
cnf_matrix = metrics.confusion_matrix(X_test_df_result['Actuals'], X_test_df_result['TARGET'])
np.set_printoptions(precision=2)
# plt.figure()
plot_confusion_matrix(cnf_matrix, 
                    classes=["Satisfied", "Un-satified"],
                    title='Confusion matrix',
                     normalize=False)
# X_test_df_result[(X_test_df_result['TARGET'] == 1) & (X_test_df_result['Actuals'] == 1)]
df_to_submit.to_csv('csv_to_submit.csv', index = False)