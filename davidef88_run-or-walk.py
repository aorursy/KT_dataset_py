import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

pl.style.use("bmh")

%matplotlib inline
d = pd.read_csv("../input/dataset.csv")
print(d.head())
print(d.dtypes)
d.date = pd.to_datetime(d.date)
d.time = np.array([a[::-1].replace(":",".",1)[::-1] for a in d.time.values], dtype=object)
d.time = pd.to_timedelta(d.time)
d.date += d.time
d = d.drop("time", axis=1)
print(d.username.unique())
d = d.drop("username", axis=1)
pl.figure(figsize=(18, 10))

pl.subplot(2,4,1)
tmp = pl.bar([0, 1], [np.sum(d.wrist.values == i) for i in range(2)], 0.8)
pl.xlabel("Wrist")
pl.ylabel("Counts")

pl.subplot(2,4,2)
tmp = pl.hist(d.acceleration_x[d.activity == 1], bins=20, range=[-5, 5], label="Run", alpha=0.5)
tmp = pl.hist(d.acceleration_x[d.activity == 0], bins=20, range=[-5, 5], label="Walk", alpha=0.5)
pl.legend()
pl.xlabel("Accel. x")

pl.subplot(2,4,3)
tmp = pl.hist(d.acceleration_y[d.activity == 1], bins=20, range=[-5, 5], label="Run", alpha=0.5)
tmp = pl.hist(d.acceleration_y[d.activity == 0], bins=20, range=[-5, 5], label="Walk", alpha=0.5)
pl.legend()
pl.xlabel("Accel. y")

pl.subplot(2,4,4)
tmp = pl.hist(d.acceleration_z[d.activity == 1], bins=20, range=[-5, 5], label="Run", alpha=0.5)
tmp = pl.hist(d.acceleration_z[d.activity == 0], bins=20, range=[-5, 5], label="Walk", alpha=0.5)
pl.legend()
pl.xlabel("Accel. z")

pl.subplot(2,4,5)
tmp = pl.bar([0, 1], [np.sum(d.activity.values == i) for i in range(2)], 0.8)
pl.xlabel("Activity")
pl.ylabel("Counts")

pl.subplot(2,4,6)
tmp = pl.hist(d.gyro_x[d.activity == 1], bins=20, range=[-5, 5], label="Run", alpha=0.5)
tmp = pl.hist(d.gyro_x[d.activity == 0], bins=20, range=[-5, 5], label="Walk", alpha=0.5)
pl.legend()
pl.xlabel("Gyro. x")

pl.subplot(2,4,7)
tmp = pl.hist(d.gyro_y[d.activity == 1], bins=20, range=[-5, 5], label="Run", alpha=0.5)
tmp = pl.hist(d.gyro_y[d.activity == 0], bins=20, range=[-5, 5], label="Walk", alpha=0.5)
pl.legend()
pl.xlabel("Gyro. y")

pl.subplot(2,4,8)
tmp = pl.hist(d.gyro_z[d.activity == 1], bins=20, range=[-5, 5], label="Run", alpha=0.5)
tmp = pl.hist(d.gyro_z[d.activity == 0], bins=20, range=[-5, 5], label="Walk", alpha=0.5)
pl.legend()
pl.xlabel("Gyro. z")

pl.subplots_adjust(wspace=0.3)
pl.figure(figsize=(18, 10))

pl.subplot(1,3,1)
tmp = pl.hist(d.acceleration_x[d.wrist == 0], bins=20, range=[-5, 5], label="Left", normed=True, alpha=0.5)
tmp = pl.hist(d.acceleration_x[d.wrist == 1], bins=20, range=[-5, 5], label="Right", normed=True, alpha=0.5)
pl.legend()
pl.xlabel("Accel. x")
pl.ylabel("Norm. Counts")

pl.subplot(1,3,2)
tmp = pl.hist(d.acceleration_y[d.wrist == 1], bins=20, range=[-5, 5], label="Right", normed=True, alpha=0.5)
tmp = pl.hist(d.acceleration_y[d.wrist == 0], bins=20, range=[-5, 5], label="Left", normed=True, alpha=0.5)
pl.legend()
pl.xlabel("Accel. y")

pl.subplot(1,3,3)
tmp = pl.hist(d.acceleration_z[d.wrist == 1], bins=20, range=[-5, 5], label="Right", normed=True, alpha=0.5)
tmp = pl.hist(d.acceleration_z[d.wrist == 0], bins=20, range=[-5, 5], label="Left", normed=True, alpha=0.5)
pl.legend()
pl.xlabel("Accel. z")

pl.subplots_adjust(wspace=0.3)
start_date = d.date.dt.date.unique()[3]
end_date = start_date + pd.Timedelta(days=1)
dd = d[np.logical_and(d.date > start_date, d.date < end_date)]


pl.figure(figsize=(18, 10))

pl.subplot(2,4,1)
pl.plot(dd.date.dt.time[dd.activity == 1], dd.acceleration_x[dd.activity == 1], "-", label="Run")
pl.plot(dd.date.dt.time[dd.activity == 0], dd.acceleration_x[dd.activity == 0], "-", label="Walk")
pl.legend()
pl.xlabel("Time - %s"%(start_date))
pl.ylabel("Accel. x")

pl.subplot(2,4,2)
pl.plot(dd.date.dt.time[dd.activity == 1], dd.acceleration_y[dd.activity == 1], "-", label="Run")
pl.plot(dd.date.dt.time[dd.activity == 0], dd.acceleration_y[dd.activity == 0], "-", label="Walk")
pl.legend()
pl.xlabel("Time - %s"%(start_date))
pl.ylabel("Accel. y")

pl.subplot(2,4,3)
pl.plot(dd.date.dt.time[dd.activity == 1], dd.acceleration_z[dd.activity == 1], "-", label="Run")
pl.plot(dd.date.dt.time[dd.activity == 0], dd.acceleration_z[dd.activity == 0], "-", label="Walk")
pl.legend()
pl.xlabel("Time - %s"%(start_date))
pl.ylabel("Accel. z")

pl.subplot(2,4,4)

a_run = np.sqrt(dd.acceleration_x[dd.activity == 1]**2 + \
                dd.acceleration_y[dd.activity == 1]**2 + \
                dd.acceleration_z[dd.activity == 1]**2)
a_walk = np.sqrt(dd.acceleration_x[dd.activity == 0]**2 + \
                 dd.acceleration_y[dd.activity == 0]**2 + \
                 dd.acceleration_z[dd.activity == 0]**2)

pl.plot(dd.date.dt.time[dd.activity == 1], a_run, "-", label="Run")
pl.plot(dd.date.dt.time[dd.activity == 0], a_walk, "-", label="Walk")
pl.legend()
pl.xlabel("Time - %s"%(start_date))
pl.ylabel("Acceleration magn.")


pl.subplot(2,4,5)
pl.plot(dd.date.dt.time[dd.activity == 1], dd.gyro_x[dd.activity == 1], "-", label="Run")
pl.plot(dd.date.dt.time[dd.activity == 0], dd.gyro_x[dd.activity == 0], "-", label="Walk")
pl.legend()
pl.xlabel("Time - %s"%(start_date))
pl.ylabel("Gyro. x")

pl.subplot(2,4,6)
pl.plot(dd.date.dt.time[dd.activity == 1], dd.gyro_y[dd.activity == 1], "-", label="Run")
pl.plot(dd.date.dt.time[dd.activity == 0], dd.gyro_y[dd.activity == 0], "-", label="Walk")
pl.legend()
pl.xlabel("Time - %s"%(start_date))
pl.ylabel("Gyro. y")

pl.subplot(2,4,7)
pl.plot(dd.date.dt.time[dd.activity == 1], dd.gyro_z[dd.activity == 1], "-", label="Run")
pl.plot(dd.date.dt.time[dd.activity == 0], dd.gyro_z[dd.activity == 0], "-", label="Walk")
pl.legend()
pl.xlabel("Time - %s"%(start_date))
pl.ylabel("Gyro. z")

pl.subplots_adjust(wspace=0.2)
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

X = d.as_matrix(columns=["acceleration_x", "acceleration_y", "acceleration_z", "gyro_x", "gyro_y", "gyro_z"])
X = preprocessing.scale(X)
Y = d.activity.values

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=1988)
from sklearn.svm import SVC

classifier_svm = SVC(kernel="rbf", random_state=2018)

#param_grid = {"C" : np.logspace(0, 3, 18)}
param_grid = {"C" : [160., 170., 180., 190., 200., 210., 220.]}

NJobs = 4

cv = model_selection.GridSearchCV(classifier_svm, param_grid=param_grid, scoring="f1", n_jobs=NJobs)
cv.fit(X_train, Y_train)
print("CV approx. total time : ", np.sum(cv.cv_results_["mean_fit_time"] + cv.cv_results_["mean_score_time"])*cv.n_splits_ / 60 / NJobs, "min")
print("Best value of C after 3-fold CV : ", cv.best_params_["C"])

classifier_svm.set_params(**cv.best_params_)
classifier_svm.fit(X_train, Y_train)
y_pred = classifier_svm.predict(X_test)
y_score = classifier_svm.decision_function(X_test)

print("Accuracy score on test set: ", metrics.accuracy_score(Y_test, y_pred))
print("F1 metric score on test set: ", metrics.f1_score(Y_test, y_pred))

# Calculate ROC and associated AUC

fpr, tpr, _ = metrics.roc_curve(Y_test, y_score)
auc = metrics.auc(fpr, tpr)

# Plot ROC
pl.figure(figsize=(6,6))
pl.subplot(1,1,1)

pl.plot([0, 1], [0, 1], ":k", lw=1)
pl.plot(fpr, tpr, "-", c="darkorange", lw=2, drawstyle="steps-pre")
pl.text(0.75, 0.2, "AUC = %.3f"%(auc))

pl.title("SVM classifier ROC curve")
pl.xlim([-0.01,1.01])
pl.ylim([-0.01,1.01])
pl.xlabel("False positive rate")
pl.ylabel("True positive rate")
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

features_to_use = [0, 1, 2]

# Initialise and try first the classifier
classifier_dt = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=1988)
classifier_dt.fit(X_train[:, features_to_use], Y_train)

print("F1 score with max_depth=3 : ", metrics.f1_score(Y_test, classifier_dt.predict(X_test[:, features_to_use])))

# Cross-validation search of max_depth and criterion

param_grid = {"criterion" : ["gini", "entropy"], "max_depth" : np.arange(2,12,1, dtype=np.int32)}
NJobs = 4

cv = model_selection.GridSearchCV(classifier_dt, param_grid=param_grid, scoring="f1", n_jobs=NJobs, cv=5)
cv.fit(X_train[:, features_to_use], Y_train)

print("Best model from CV : ", cv.best_params_, ", F1 score : ", cv.best_score_)
classifier_dt.set_params(**cv.best_params_)
classifier_dt.fit(X_train[:, features_to_use], Y_train)

for i,feat in enumerate(["accel. x", "accel. y", "accel. z"]):
    print("- Importance of feature", feat, " : ", classifier_dt.feature_importances_[i])
y_pred = classifier_dt.predict(X_test[:, features_to_use])
y_score = classifier_dt.predict_proba(X_test[:, features_to_use])[:,1]

print("Accuracy score on test set: ", metrics.accuracy_score(Y_test, y_pred))
print("F1 metric score on test set: ", metrics.f1_score(Y_test, y_pred))

# Calculate ROC and associated AUC

fpr, tpr, _ = metrics.roc_curve(Y_test, y_score)
auc = metrics.auc(fpr, tpr)

# Plot ROC
pl.figure(figsize=(6,6))
pl.subplot(1,1,1)

pl.plot([0, 1], [0, 1], ":k", lw=1)
pl.plot(fpr, tpr, "-", c="darkorange", lw=2, drawstyle="steps-pre")
pl.text(0.75, 0.2, "AUC = %.3f"%(auc))

pl.title("DT classifier ROC curve")
pl.xlim([-0.01,1.01])
pl.ylim([-0.01,1.01])
pl.xlabel("False positive rate")
pl.ylabel("True positive rate")
classifier_rf = RandomForestClassifier(criterion="gini", max_depth=9, random_state=2018, oob_score=True)

#param_grid = {"n_estimators" : np.logspace(1.0, 2.0, 6, dtype=np.int32)}
param_grid = {"n_estimators" : np.arange(50, 101, 10, dtype=np.int32)}
NJobs = 4

cv = model_selection.GridSearchCV(classifier_rf, param_grid=param_grid, scoring="f1", n_jobs=NJobs, cv=3)
cv.fit(X_train[:, features_to_use], Y_train)

print("Best model from CV : ", cv.best_params_, ", OOB score : ", cv.best_score_)
classifier_rf.set_params(**cv.best_params_)
classifier_rf.fit(X_train[:, features_to_use], Y_train)

for i,feat in enumerate(["accel. x", "accel. y", "accel. z"]):
    print("- Importance of feature", feat, " : ", classifier_rf.feature_importances_[i])
y_pred = classifier_rf.predict(X_test[:, features_to_use])
y_score = classifier_rf.predict_proba(X_test[:, features_to_use])[:,1]

print("Accuracy score on test set: ", metrics.accuracy_score(Y_test, y_pred))
print("F1 metric score on test set: ", metrics.f1_score(Y_test, y_pred))

# Calculate ROC and associated AUC

fpr, tpr, _ = metrics.roc_curve(Y_test, y_score)
auc = metrics.auc(fpr, tpr)

# Plot ROC
pl.figure(figsize=(6,6))
pl.subplot(1,1,1)

pl.plot([0, 1], [0, 1], ":k", lw=1)
pl.plot(fpr, tpr, "-", c="darkorange", lw=2, drawstyle="steps-pre")
pl.text(0.75, 0.2, "AUC = %.3f"%(auc))

pl.title("DT classifier ROC curve")
pl.xlim([-0.01,1.01])
pl.ylim([-0.01,1.01])
pl.xlabel("False positive rate")
pl.ylabel("True positive rate")
clf = [
    SVC(C=180, kernel="rbf", random_state=2018),
    DecisionTreeClassifier(criterion="gini", max_depth=9, random_state=1988),
    RandomForestClassifier(criterion="gini", max_depth=9, n_estimators=50, random_state=1988)
]

pl.figure(figsize=(15, 15))

x1, x2 = np.meshgrid(np.arange(-5,5,0.1), np.arange(-5,5,0.1), indexing="ij")

comb_feat = [[0,1], [0,2], [1,2]]
label_feat = [["Accel. x", "Accel. y"], ["Accel. x", "Accel. z"], ["Accel. y", "Accel. z"]]
title = ["SVC", "DecisionTree", "RandomForest", "kNN"]
for i in range(3):
    for j in range(3):
        pl.subplot(3, 3, i*3 + j + 1)
        
        clf[j].fit(X_train[::100, comb_feat[i]], Y_train[::100])
        
        if hasattr(clf[j], "decision_function"):
            Z = clf[j].decision_function(np.array([x1.ravel(), x2.ravel()]).T)
        else:
            Z = clf[j].predict_proba(np.array([x1.ravel(), x2.ravel()]).T)[:, 1]
    
        Z = Z.reshape(x1.shape)
        pl.contourf(x1, x2, Z, cmap="RdBu", alpha=.8)
    
        pl.plot(X_test[Y_test == 1, comb_feat[i][0]][::100], X_test[Y_test == 1, comb_feat[i][1]][::100], "ob", alpha=0.4)
        pl.plot(X_test[Y_test == 0, comb_feat[i][0]][::100], X_test[Y_test == 0, comb_feat[i][1]][::100], "or", alpha=0.4)
    
        pl.xlim([-5, 5])
        pl.ylim([-5, 5])
        pl.xlabel(label_feat[i][0])
        pl.ylabel(label_feat[i][1])
        
        if i == 0:
            pl.title(title[j])
    
    print("Trained row", i+1)

pl.subplots_adjust(wspace=0.25, hspace=0.25)
