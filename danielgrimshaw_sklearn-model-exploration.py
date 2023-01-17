import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



prefix = "../input/"
df = pd.read_csv(prefix + "train.csv", index_col='ID_code')

trues = df.loc[df['target'] == 1]

falses = df.loc[df['target'] != 1].sample(frac=1)[:len(trues)]

data = pd.concat([trues, falses], ignore_index=True).sample(frac=1)

data.head()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



y = data['target']

X = data.drop('target', axis=1)



scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.linear_model import RidgeClassifier, SGDClassifier

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC, NuSVC, SVC

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier



models = []

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('QDA', QuadraticDiscriminantAnalysis()))

models.append(('AdaBoost', AdaBoostClassifier()))

models.append(('Bagging', BaggingClassifier()))

models.append(('Extra Trees Ensemble', ExtraTreesClassifier(n_estimators=1000)))

models.append(('Gradient Boosting', GradientBoostingClassifier()))

models.append(('Random Forest', RandomForestClassifier(n_estimators=1000)))

models.append(('Ridge', RidgeClassifier()))

models.append(('SGD', SGDClassifier(tol=1e-3, max_iter=10000)))

models.append(('BNB', BernoulliNB()))

models.append(('GNB', GaussianNB()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('MLP', MLPClassifier()))

models.append(('LSVC', LinearSVC(max_iter=100000)))

models.append(('NuSVC', NuSVC(gamma='scale')))

models.append(('SVC', SVC(gamma='scale')))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('ETC', ExtraTreeClassifier()))



DECISION_FUNCTIONS = {"Ridge", "SGD", "LSVC", "NuSVC", "SVC"}
from sklearn.metrics import roc_auc_score, roc_curve

%matplotlib inline



best_model = None

best_model_name = ""

best_valid = 0

for name, model in models:

    model.fit(X_train, y_train)

    if name in DECISION_FUNCTIONS:

        proba = model.decision_function(X_valid)

    else:

        proba = model.predict_proba(X_valid)[:, 1]

    score = roc_auc_score(y_valid, proba)

    fpr, tpr, _  = roc_curve(y_valid, proba)

    plt.figure()

    plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (auc = {score})")

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.title(f"{name} Results")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right")

    plt.show()

    if score > best_valid:

        best_valid = score

        best_model = model

        best_model_name = name



print(f"Best model is {best_model_name}")
test = pd.read_csv(prefix + "test.csv", index_col='ID_code')

submission = pd.read_csv(prefix + "sample_submission.csv", index_col='ID_code')



test_X = scaler.transform(test)

if best_model_name in DECISION_FUNCTIONS:

    submission['target'] = best_model.decision_function(test_X)

else:

    submission['target'] = best_model.predict_proba(test_X)[:, 1]

submission.to_csv(f"{best_model_name}_submission.csv")