# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv", sep = ",")
data.head()
data.info()
data.isnull().any()
data.describe()
fraud = data[data.Class == 1]
normal = data[data.Class == 0]
fraud.shape
normal.shape
fraud.Amount.describe()
normal.Amount.describe()
import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize = (15,15))
f.suptitle('Amount per transaction by class')

bins = 50

ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')

ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize = (10,15))
f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')

ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
from sklearn.preprocessing import StandardScaler
df = data.drop(['Time'], axis=1)
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df.head()
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:29],df[['Class']], test_size = 0.25, random_state = 42)

xgboost_model = xgboost.XGBClassifier()

parameters = {
        'n_estimators': [10,50,100,200],
        'subsample': [0.1,0.5,0.8], 
        'gamma' : [0,1,3,5],
        'max_depth': [3, 4, 5], 
        'learning_rate': [0.01,0.1,0.3]}
df.iloc[:,:29]
xgboost_cv = GridSearchCV(xgboost_model, parameters, cv = 3, n_jobs = -1, verbose = 2)
xgboost_cv.fit(X_train, y_train)
best = xgboost_cv.best_params_
best
best_params = {'objective':'binary:logistic',
 'learning_rate': 0.1,
 'max_depth': 5,
 'subsample': 0.8,
 'n_estimators': 200}
xgboost_classifier = xgboost.XGBClassifier(**best_params)
xgb_tuned =  xgboost_classifier.fit(X_train,y_train)
evals_result = xgb_tuned.evals_result
evals_result
xgb_tuned.predict(X_test)
xgb_tuned.predict_proba(X_test)
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, precision_recall_curve
xgdmat_train = xgboost.DMatrix(X_train,y_train)
xgdmat_test = xgboost.DMatrix(X_test, y_test)
xgb_final = xgboost.train(best_params, xgdmat_train, num_boost_round = 100)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def show_data(cm, print_res = 0):
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    if print_res == 1:
        print('Precision =     {:.3f}'.format(tp/(tp+fp)))
        print('Recall (TPR) =  {:.3f}'.format(tp/(tp+fn)))
        print('Fallout (FPR) = {:.3e}'.format(fp/(fp+tn)))
    return tp/(tp+fp), tp/(tp+fn), fp/(fp+tn)
import itertools

y_pred = xgb_final.predict(xgdmat_test)
thresh = 0.08
y_pred [y_pred > thresh] = 1
y_pred [y_pred <= thresh] = 0
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, ['0', '1'], )
pr, tpr, fpr = show_data(cm, print_res = 1);
