import sklearn.model_selection

from sklearn import svm

import pandas as pd

import numpy as np

import datetime as dt

import seaborn as sns

from matplotlib import pyplot as plt

import os



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix,  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score



dirname = os.path.abspath(os.getcwd())

hard_filename = 'training_dataset_hard_openapp_intern_2020-02-25.csv'



hard_raw_data = pd.read_csv(os.path.join(dirname[:-5]+"\\dataset", hard_filename), index_col=0)



features = ['feature_1', 'feature_2','feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9']



hard_x_wo5 = hard_raw_data[features]

hard_y = hard_raw_data.churn



hard_x_wo5_train, hard_x_wo5_val, hard_y_wo5_train, hard_y_wo5_val = train_test_split(hard_x_wo5, hard_y, test_size=0.3, shuffle=False)
scores = []

pars = []
scaler = StandardScaler()

hard_x_wo5_train = scaler.fit_transform(hard_x_wo5_train)

hard_x_wo5_val = scaler.transform(hard_x_wo5_val)
def metric_score(y_true, y_pred, title=""):

    accuracy = accuracy_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    

    scores.append((accuracy, auc, f1, precision, recall))

    

    print(title)

    print("Accuracy :", accuracy)

    print("A.U.C    :", auc)

    print("F1       :", f1)

    print("Precision:", precision)

    print("Recall   :", recall)

    

def plot_confusion_matrix(y_true, y_pred, title=""):

    data = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))

    print(df_cm)

    df_cm.index.name = 'Actual'

    df_cm.columns.name = 'Predicted'

    plt.figure(figsize = (10,7))

    plt.title(title)

    sns.set(font_scale=1.4)#for label size

    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size

    plt.show()
C = [0.005, 0.05, 0.5, 1.0]

print(C)

print(len(C))
coef0 = [0.005, 0.05, 0.5, 1.0]

print(coef0)

print(len(coef0))
degree = [3]

print(degree)

print(len(degree))
CLH = [0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005, 0.00055, 0.0006, 0.00065, 0.0007, 0.00075]

print(CLH)

print(len(CLH))
%%time

kernel = ['linear']



parameters = {'kernel':kernel, 'max_iter': [-1], 'random_state' : [0], 'shrinking' : [False], 'C':CLH}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters, scoring='f1', n_jobs=-1, cv=4, verbose=20, refit=False)

clf.fit(hard_x_wo5_train, hard_y_wo5_train)

    

clf.best_score_
clf.best_params_
pars.append(clf.best_params_)
print("Now testing the best parameter")

cls_sig = svm.SVC(**clf.best_params_)
%%time

cls_sig.fit(hard_x_wo5_train, hard_y_wo5_train)
%%time

hard_y_wo5_prediction_sig = cls_sig.predict(hard_x_wo5_val)
metric_score(hard_y_wo5_val, hard_y_wo5_prediction_sig)

plot_confusion_matrix(hard_y_wo5_val, hard_y_wo5_prediction_sig)
coef0PolyH = [0.035, 0.04, 0.045]

print(coef0PolyH)

print(len(coef0PolyH))
CPolyH = [0.55, 0.6, 0.65]

print(CPolyH)

print(len(CPolyH))
%%time

kernel = ['poly']



parameters = {'kernel':kernel, 'max_iter': [-1], 'random_state' : [0], 'shrinking' : [False], 'C':CPolyH, 'coef0' : coef0PolyH, 'degree' : degree}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters, scoring='f1', cv=4, verbose=20, refit=False, n_jobs=-1)

clf.fit(hard_x_wo5_train, hard_y_wo5_train)



clf.best_score_
clf.best_params_
pars.append(clf.best_params_)
print("Now testing the best parameter")

cls_sig = svm.SVC(**clf.best_params_)
%%time

cls_sig.fit(hard_x_wo5_train, hard_y_wo5_train)
%%time

hard_y_wo5_prediction_sig = cls_sig.predict(hard_x_wo5_val)
metric_score(hard_y_wo5_val, hard_y_wo5_prediction_sig)

plot_confusion_matrix(hard_y_wo5_val, hard_y_wo5_prediction_sig)
CRH = [0.0625, 0.065, 0.0675, 0.07, 0.0725, 0.075]

print(CRH)

print(len(CRH))
%%time

kernel = ['rbf']



parameters = {'kernel':kernel, 'max_iter': [-1], 'random_state' : [0], 'shrinking' : [False], 'C':CRH}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters, scoring='f1', cv=4, verbose=20, refit=False, n_jobs=-1)

clf.fit(hard_x_wo5_train, hard_y_wo5_train)



clf.best_score_
clf.best_params_
pars.append(clf.best_params_)
print("Now testing the best parameter")

cls_sig = svm.SVC(**clf.best_params_)
%%time

cls_sig.fit(hard_x_wo5_train, hard_y_wo5_train)
%%time

hard_y_wo5_prediction_sig = cls_sig.predict(hard_x_wo5_val)
metric_score(hard_y_wo5_val, hard_y_wo5_prediction_sig)

plot_confusion_matrix(hard_y_wo5_val, hard_y_wo5_prediction_sig)
coef0SH = [0.035, 0.04, 0.045]

print(coef0SH)

print(len(coef0SH))
CSH = [0.00375, 0.004, 0.00425, 0.00475]

print(CSH)

print(len(CSH))
%%time

kernel = ['sigmoid']



parameters = {'kernel':kernel, 'max_iter': [-1], 'random_state' : [0], 'shrinking' : [False], 'C':CSH, 'coef0' : coef0SH}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters, scoring='f1', cv=4, n_jobs=-1, verbose=20, refit=False)

clf.fit(hard_x_wo5_train, hard_y_wo5_train)



clf.best_score_
clf.best_params_
pars.append(clf.best_params_)
print("Now testing the best parameter")

cls_sig = svm.SVC(**clf.best_params_)
%%time

cls_sig.fit(hard_x_wo5_train, hard_y_wo5_train)
%%time

hard_y_wo5_prediction_sig = cls_sig.predict(hard_x_wo5_val)
metric_score(hard_y_wo5_val, hard_y_wo5_prediction_sig)

plot_confusion_matrix(hard_y_wo5_val, hard_y_wo5_prediction_sig)
CLSVCH = [0.25, 0.275, 0.285, 0.3, 0.315, 0.325, 0.35]

print(CLSVCH)

print(len(CLSVCH))
max_iterLSVCH = [1200, 1300, 1400, 1500, 1600, 1700, 1800]

print(max_iterLSVCH)

print(len(max_iterLSVCH))
%%time

parameters = {'max_iter': max_iterLSVCH, 'random_state' : [0], 'C':CLSVCH}

svc = svm.LinearSVC()

clf = GridSearchCV(svc, parameters, scoring='f1', cv=4, n_jobs=-1, verbose=20, refit=False)

clf.fit(hard_x_wo5_train, hard_y_wo5_train)



clf.best_score_
clf.best_params_
pars.append(clf.best_params_)
print("Now testing the best parameter")

cls_sig = svm.LinearSVC(**clf.best_params_)
%%time

cls_sig.fit(hard_x_wo5_train, hard_y_wo5_train)
%%time

hard_y_wo5_prediction_sig = cls_sig.predict(hard_x_wo5_val)
metric_score(hard_y_wo5_val, hard_y_wo5_prediction_sig)

plot_confusion_matrix(hard_y_wo5_val, hard_y_wo5_prediction_sig)
hard_filename = 'training_dataset_soft_openapp_intern_2020-02-25.csv'



hard_raw_data = pd.read_csv(os.path.join(dirname[:-5]+"\\dataset", hard_filename), index_col=0)

hard_raw_data.dropna(inplace=True)

features = ['feature_1', 'feature_2','feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9']



hard_x_wo5 = hard_raw_data[features]

hard_y = hard_raw_data.churn



hard_x_wo5_train, hard_x_wo5_val, hard_y_wo5_train, hard_y_wo5_val = train_test_split(hard_x_wo5, hard_y, test_size=0.3, shuffle=False)
scaler = StandardScaler()

hard_x_wo5_train = scaler.fit_transform(hard_x_wo5_train)

hard_x_wo5_val = scaler.transform(hard_x_wo5_val)
CLS = [0.002, 0.00225, 0.0024, 0.0025, 0.0026, 0.00275]

print(CLS)

print(len(CLS))
%%time

kernel = ['linear']



parameters = {'kernel':kernel, 'max_iter': [-1], 'random_state' : [0], 'shrinking' : [False], 'C':CLS}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters, scoring='f1', n_jobs=-1, cv=4, verbose=20, refit=False)

clf.fit(hard_x_wo5_train, hard_y_wo5_train)



clf.best_score_
clf.best_params_
pars.append(clf.best_params_)
print("Now testing the best parameter")

cls_sig = svm.SVC(**clf.best_params_)
%%time

cls_sig.fit(hard_x_wo5_train, hard_y_wo5_train)
%%time

hard_y_wo5_prediction_sig = cls_sig.predict(hard_x_wo5_val)
metric_score(hard_y_wo5_val, hard_y_wo5_prediction_sig)

plot_confusion_matrix(hard_y_wo5_val, hard_y_wo5_prediction_sig)
CPS = [0.055, 0.06, 0.065]

print(CPS)

print(len(CPS))
coef0PS = [1.5, 2, 2.5]

print(coef0PS)

print(len(coef0PS))
%%time

kernel = ['poly']



parameters = {'kernel':kernel, 'max_iter': [-1], 'random_state' : [0], 'shrinking' : [False], 'C':CPS, 'coef0' : coef0PS, 'degree' : degree}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters, scoring='f1', cv=4, verbose=20, refit=False, n_jobs=-1)

clf.fit(hard_x_wo5_train, hard_y_wo5_train)



clf.best_score_
clf.best_params_
pars.append(clf.best_params_)
print("Now testing the best parameter")

cls_sig = svm.SVC(**clf.best_params_)
%%time

cls_sig.fit(hard_x_wo5_train, hard_y_wo5_train)
%%time

hard_y_wo5_prediction_sig = cls_sig.predict(hard_x_wo5_val)
metric_score(hard_y_wo5_val, hard_y_wo5_prediction_sig)

plot_confusion_matrix(hard_y_wo5_val, hard_y_wo5_prediction_sig)
CRS = [1.15, 1.2, 1.25, 1.5, 1.75, 2]

print(CRS)

print(len(CRS))
%%time

kernel = ['rbf']



parameters = {'kernel':kernel, 'max_iter': [-1], 'random_state' : [0], 'shrinking' : [False], 'C':CRS}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters, scoring='f1', cv=4, verbose=20, refit=False, n_jobs=-1)

clf.fit(hard_x_wo5_train, hard_y_wo5_train)



clf.best_score_
clf.best_params_
pars.append(clf.best_params_)
print("Now testing the best parameter")

cls_sig = svm.SVC(**clf.best_params_)
%%time

cls_sig.fit(hard_x_wo5_train, hard_y_wo5_train)
%%time

hard_y_wo5_prediction_sig = cls_sig.predict(hard_x_wo5_val)
metric_score(hard_y_wo5_val, hard_y_wo5_prediction_sig)

plot_confusion_matrix(hard_y_wo5_val, hard_y_wo5_prediction_sig)
coef0SS = [0.055, 0.06, 0.065]

print(coef0SS)

print(len(coef0SS))
CSS = [0.0035, 0.004, 0.0045]

print(CSS)

print(len(CSS))
%%time

kernel = ['sigmoid']



parameters = {'kernel':kernel, 'max_iter': [-1], 'random_state' : [0], 'shrinking' : [False], 'C':CSS, 'coef0' : coef0SS}

svc = svm.SVC()

clf = GridSearchCV(svc, parameters, scoring='f1', cv=4, n_jobs=-1, verbose=20, refit=False)

clf.fit(hard_x_wo5_train, hard_y_wo5_train)



clf.best_score_
clf.best_params_
pars.append(clf.best_params_)
print("Now testing the best parameter")

cls_sig = svm.SVC(**clf.best_params_)
%%time

cls_sig.fit(hard_x_wo5_train, hard_y_wo5_train)
%%time

hard_y_wo5_prediction_sig = cls_sig.predict(hard_x_wo5_val)
metric_score(hard_y_wo5_val, hard_y_wo5_prediction_sig)

plot_confusion_matrix(hard_y_wo5_val, hard_y_wo5_prediction_sig)
CLSVCS = [0.325, 0.33, 0.34, 0.345, 0.35, 0.355, 0.36, 0.375]

print(CLSVCS)

print(len(CLSVCS))
max_iterLSVCS = [1200, 1300, 1400, 1500, 1600, 1700, 1800]

print(max_iterLSVCS)

print(len(max_iterLSVCS))
%%time

parameters = {'max_iter': max_iterLSVCS, 'random_state' : [0], 'C':CLSVCS}

svc = svm.LinearSVC()

clf = GridSearchCV(svc, parameters, scoring='f1', cv=4, n_jobs=-1, verbose=20, refit=False)

clf.fit(hard_x_wo5_train, hard_y_wo5_train)



clf.best_score_
clf.best_params_
pars.append(clf.best_params_)
print("Now testing the best parameter")

cls_sig = svm.LinearSVC(**clf.best_params_)
%%time

cls_sig.fit(hard_x_wo5_train, hard_y_wo5_train)
%%time

hard_y_wo5_prediction_sig = cls_sig.predict(hard_x_wo5_val)
metric_score(hard_y_wo5_val, hard_y_wo5_prediction_sig)

plot_confusion_matrix(hard_y_wo5_val, hard_y_wo5_prediction_sig)
scores
pars