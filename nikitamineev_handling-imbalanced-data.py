# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score, plot_confusion_matrix

from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def get_result_table(y_true, Y_pred, LABELS=None, metrics=None):
    if metrics is None: metrics = [accuracy_score, f1_score, roc_auc_score]
    RES = pd.DataFrame(index=LABELS)
    
    for metric in metrics:
        RES[metric.__name__] = [np.round(metric(y_true, y_pred), 3) for y_pred in Y_pred]
    
    return RES
data = pd.read_csv(r"/kaggle/input/creditcardfraud/creditcard.csv")
#data reduction for faster fitting
data = pd.concat([data[data["Class"] == 0].sample(9000), data[data["Class"] == 1]]).sample(frac=1).reset_index(drop=True)
#splitting data
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data["Class"], test_size=0.2)
print(data["Class"].value_counts())
sns.countplot(data['Class'])
svm_mdl = LinearSVC().fit(X_train, y_train)
nn_mdl = MLPClassifier(max_iter=300).fit(X_train, y_train)
xgb_mdl = XGBClassifier().fit(X_train, y_train)

Y_pred = [mdl.predict(X_test) for mdl in [svm_mdl, nn_mdl, xgb_mdl]]
get_result_table(y_test, Y_pred, LABELS=["SVM", "NN", "XGB"])
X_train_run, y_train_run = RandomUnderSampler().fit_sample(X_train,y_train)
print(y_train_run.value_counts())
sns.countplot(y_train_run)
svm_mdl_run = LinearSVC().fit(X_train_run, y_train_run)
nn_mdl_run = MLPClassifier(max_iter=300).fit(X_train_run, y_train_run)
xgb_mdl_run = XGBClassifier().fit(X_train_run, y_train_run)

Y_pred_run = [mdl.predict(X_test) for mdl in [svm_mdl_run, nn_mdl_run, xgb_mdl_run]]
get_result_table(y_test, Y_pred_run, LABELS=["SVM", "NN", "XGB"])
X_train_tl, y_train_tl = TomekLinks(sampling_strategy='majority').fit_sample(X_train,y_train)
print(y_train_tl.value_counts())
sns.countplot(y_train_tl)
svm_mdl_tl = LinearSVC().fit(X_train_tl, y_train_tl)
nn_mdl_tl = MLPClassifier(max_iter=300).fit(X_train_tl, y_train_tl)
xgb_mdl_tl = XGBClassifier().fit(X_train_tl, y_train_tl)

Y_pred_tl = [mdl.predict(X_test) for mdl in [svm_mdl_tl, nn_mdl_tl, xgb_mdl_tl]]
get_result_table(y_test, Y_pred_tl, LABELS=["SVM", "NN", "XGB"])
X_train_rov, y_train_rov = RandomOverSampler().fit_sample(X_train,y_train)
print(y_train_rov.value_counts())
sns.countplot(y_train_rov)
svm_mdl_rov = LinearSVC().fit(X_train_rov, y_train_rov)
nn_mdl_rov = MLPClassifier(max_iter=300).fit(X_train_rov, y_train_rov)
xgb_mdl_rov = XGBClassifier().fit(X_train_rov, y_train_rov)

Y_pred_rov = [mdl.predict(X_test) for mdl in [svm_mdl_rov, nn_mdl_rov, xgb_mdl_rov]]
get_result_table(y_test, Y_pred_rov, LABELS=["SVM", "NN", "XGB"])
X_train_smote, y_train_smote = SMOTE(sampling_strategy='minority').fit_sample(X_train,y_train)
print(y_train_smote.value_counts())
sns.countplot(y_train_smote)
svm_mdl_smote = LinearSVC().fit(X_train_smote, y_train_smote)
nn_mdl_smote = MLPClassifier(max_iter=300).fit(X_train_smote, y_train_smote)
xgb_mdl_smote = XGBClassifier().fit(X_train_smote, y_train_smote)

Y_pred_smote = [mdl.predict(X_test) for mdl in [svm_mdl_smote, nn_mdl_smote, xgb_mdl_smote]]
get_result_table(y_test, Y_pred_smote, LABELS=["SVM", "NN", "XGB"])
