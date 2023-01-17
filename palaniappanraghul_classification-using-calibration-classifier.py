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
from sklearn import datasets

from sklearn.model_selection import train_test_split





# create a random, binary classification problem, with 100000 samples and 20 features

X, y = datasets.make_classification(n_samples=100000, n_features=20,

                                    n_informative=7, n_redundant=10,

                                    random_state=42)



# split into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.99, random_state=42)

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression





lgr = LogisticRegression(C=1, solver='lbfgs')

svc = SVC(max_iter=10000, probability=True)
probs_lgr = lgr.fit(X_train, y_train).predict_proba(X_test)[:,1]

preds_svc = svc.fit(X_train, y_train).predict(X_test)



probs_svc = svc.decision_function(X_test)

probs_svc = (probs_svc - probs_svc.min()) / (probs_svc.max() - probs_svc.min())
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(10,6))

sns.kdeplot(probs_lgr, label='Logistic regression')

sns.kdeplot(preds_svc, label='SVM')

plt.title("Probability Density Plot for 2 Classifiers")

plt.show()


from sklearn import metrics





plt.figure(figsize=(8,5))

plt.plot([0, 1], [0, 1],'r--')



pred = probs_lgr

label = y_test

fpr, tpr, thresh = metrics.roc_curve(label, pred)

auc = metrics.roc_auc_score(label, pred)

plt.plot(fpr, tpr, label=f'Logistic regression, auc = {str(round(auc,3))}')



pred = probs_svc

fpr, tpr, thresh = metrics.roc_curve(label, pred)

auc = metrics.roc_auc_score(label, pred)

plt.plot(fpr, tpr, label=f'SVC, auc = {str(round(auc,3))}')



plt.ylabel("True Positive Rate")

plt.xlabel("False Positive Rate")

plt.title("AUC-ROC for two models")

plt.legend()

plt.show()
from sklearn.calibration import calibration_curve





def plot_calibration_curve(name, fig_index, probs):

    """Plot calibration curve for est w/o and with calibration. """



    fig = plt.figure(fig_index, figsize=(10, 10))

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    ax2 = plt.subplot2grid((3, 1), (2, 0))

    

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    

    frac_of_pos, mean_pred_value = calibration_curve(y_test, probs, n_bins=10)



    ax1.plot(mean_pred_value, frac_of_pos, "s-", label=f'{name}')

    ax1.set_ylabel("Fraction of positives")

    ax1.set_ylim([-0.05, 1.05])

    ax1.legend(loc="lower right")

    ax1.set_title(f'Calibration plot ({name})')

    

    ax2.hist(probs, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax2.set_xlabel("Mean predicted value")

    ax2.set_ylabel("Count")
# plot calibration curve for logistic regression

plot_calibration_curve("Logistic regression", 1, probs_lgr)
# plot calibration curve for the SVM

plot_calibration_curve("SVM", 1, probs_svc)
from sklearn.calibration import CalibratedClassifierCV





lgr = LogisticRegression(C=1, solver='lbfgs')

svc = SVC(max_iter=10000, probability=True)



platts_scaling = CalibratedClassifierCV(svc, cv=2, method='sigmoid')

platts_scaling.fit(X_train, y_train)

calibrated_probs = platts_scaling.predict_proba(X_test)[:,1]



platts_scaling = CalibratedClassifierCV(lgr, cv=2, method='sigmoid')

platts_scaling.fit(X_train, y_train)

calibrated_probs1 = platts_scaling.predict_proba(X_test)[:,1]





plot_calibration_curve("SVM", 3, calibrated_probs)

plot_calibration_curve("Logistic regression", 1,calibrated_probs1)