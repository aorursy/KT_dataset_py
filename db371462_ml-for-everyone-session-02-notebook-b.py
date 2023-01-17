%matplotlib inline
import sys



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split, GridSearchCV





# Let's print what versions of the libraries we're using

print(f"python\t\tv {sys.version.split(' ')[0]}\n===")

for lib_ in [np, pd, sns, sklearn, ]:

    sep_ = '\t' if len(lib_.__name__) > 8 else '\t\t'

    print(f"{lib_.__name__}{sep_}v {lib_.__version__}"); del sep_, lib_
import os

os.getcwd()
!ls
hitters = pd.read_csv("../input/hitters/Hitters.csv")

hitters = hitters.dropna(inplace=False)

hitters.head()
X = np.array(hitters.drop(["Salary", "League", "Division", "NewLeague"], axis=1))

y = (hitters["Salary"] >= np.median(hitters["Salary"])).astype("int")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=10)
print(X_train.shape)

print(X_test.shape)
logit = LogisticRegression(penalty="l2", C=1e5, n_jobs=-1, max_iter=4000)

logit.fit(X_train, y_train)



test_preds = logit.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, test_preds)
# we want to draw the random baseline ROC line too

fpr_rand = tpr_rand = np.linspace(0, 1, 10)



plt.plot(fpr, tpr)

plt.plot(fpr_rand, tpr_rand, linestyle='--')

plt.show()
roc_auc_score(y_test, test_preds)
# create equally space values beteen 10^-10 and 10^10

c_vals = np.logspace(-10, 10, 20)



aucs = []

for c_val in c_vals:

    logit = LogisticRegression(C=c_val)

    logit.fit(X_train, y_train)



    test_preds = logit.predict_proba(X_test)[:, 1]

    aucs.append(roc_auc_score(y_test, test_preds))
aucs
plt.plot(np.log10(c_vals), aucs)

plt.xlabel("C")

plt.ylabel("Test AUC")

plt.show()
param_grid = {"C": np.logspace(2, 8, 50)}
cv = GridSearchCV(logit, param_grid, cv=10, n_jobs=-1, refit=True, verbose=True)

cv.fit(X_train, y_train)
cv.best_estimator_
cv.best_params_
np.log10(1.0/cv.best_params_['C'])
cv.best_score_
cv.cv_results_
test_preds = cv.best_estimator_.predict_proba(X_test)[:, 1]

test_preds
roc_auc_score(y_test, test_preds)