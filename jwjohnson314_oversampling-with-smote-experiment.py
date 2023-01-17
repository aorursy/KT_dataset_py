import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score as acc

from sklearn.metrics import classification_report, roc_curve, auc

from sklearn.ensemble import RandomForestClassifier as RF

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE



% matplotlib inline

plt.style.use('bmh')
! head ../input/exoTrain.csv | cut -d ',' -f 1-5
train_data = np.genfromtxt('../input/exoTrain.csv', delimiter=',', skip_header=1)

X = train_data[:, 1:]

y = train_data[:, 0] - 1
sum(y == 1) / len(y)
X_sm, y_sm = SMOTE(random_state=0).fit_sample(X, y)

Xtr, Xte, ytr, yte = train_test_split(X_sm, y_sm, test_size=0.2, random_state=0)
rf = RF(n_estimators=100, n_jobs=-1).fit(Xtr, ytr)

rf_preds = rf.predict(Xte)

rf_probs = rf.predict_proba(Xte)

print(acc(yte, rf_preds))
def plot_roc(actual, probs):

    tpr = dict()

    fpr = dict()

    roc_auc = dict()

    fpr, tpr, _ = roc_curve(actual, probs[:, 1])

    roc_auc = auc(fpr, tpr)

    

    plt.plot(fpr, tpr, label='AUC: {:.6f}'.format(roc_auc))

    plt.legend(loc='best')
plot_roc(yte, rf_probs)
test_data = np.genfromtxt('../input/exoTest.csv', delimiter=',', skip_header=1)

test_labels = test_data[:, 0] - 1

test_features = test_data[:, 1:]



test_preds = rf.predict(test_features)

print(acc(test_labels, test_preds))

print(classification_report(test_labels, test_preds))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

Xtr_sm , ytr_sm = SMOTE(random_state=0).fit_sample(Xtr, ytr)

Xte_sm, yte_sm = SMOTE(random_state=0).fit_sample(Xte, yte)
rf = RF(n_estimators=100, n_jobs=-1).fit(Xtr_sm, ytr_sm)

rf_preds = rf.predict(Xte_sm)

rf_probs = rf.predict_proba(Xte_sm)

print(acc(yte_sm, rf_preds))
test_data = np.genfromtxt('../input/exoTest.csv', delimiter=',', skip_header=1)

test_labels = test_data[:, 0] - 1

test_features = test_data[:, 1:]



test_preds = rf.predict(test_features)

print(acc(test_labels, test_preds))

print(classification_report(test_labels, test_preds))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

Xtr_sm, ytr_sm = SMOTE(random_state=0).fit_sample(Xtr, ytr)



rf = RF(n_estimators=100, n_jobs=-1).fit(Xtr_sm, ytr_sm)

rf_preds = rf.predict(Xte)

print(acc(yte, rf_preds))

print(classification_report(yte, rf_preds))
test_preds = rf.predict(test_features)

print(acc(test_labels, test_preds))

print(classification_report(test_labels, test_preds))