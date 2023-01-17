import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/breast-cancer-dataset/dataR2.csv")
data.info()
pd.options.display.float_format = "{:.2f}".format
data.describe()
classes = data['Classification']
ax = sns.countplot(x=classes, data=data)
sns.set_style('white')
sns.set_context('notebook')
sns.pairplot(data, hue='Classification', palette='bwr', height=2)
label_encoder = LabelEncoder()
data['Classification'] = label_encoder.fit_transform(data['Classification'])
data.head()
corr = data.corr()
plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot= True)
X = data.drop(['Classification'], axis=1)
y = data['Classification'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
classes_test=pd.DataFrame(y_test.reshape(-1,1))
classes_test[0].value_counts()
def plot_roc(roc_auc, false_positive_rate, true_positive_rate):
  plt.figure(figsize=(6, 6))
  plt.title('Receiver Operating Characteristics')
  plt.plot(false_positive_rate, true_positive_rate, color='red', label='AUC = {:.2f}'.format( roc_auc))
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.axis('tight')
  plt.ylabel('True Positive Rtae')
  plt.xlabel('False Positive Rtae')
solvers = ['svd', 'lsqr', 'eigen']
parameters = dict(solver=solvers)
lda = GridSearchCV(
    LinearDiscriminantAnalysis(), parameters, cv=5,scoring='accuracy'
    )
lda.fit(X, y.ravel())
lda_opt = lda.best_estimator_
print(lda.best_params_)
print(lda.best_score_)
lda = LinearDiscriminantAnalysis(solver='lsqr')
lda.fit(X_train, y_train.ravel())
lda_pred = lda.predict(X_test)
metrics.accuracy_score(lda_pred, y_test)
confusion_matrix = metrics.confusion_matrix(y_test, lda_pred)
confusion_matrix
false_positive_rate_lda, true_positive_rate_lda, thresholds = metrics.roc_curve(
    y_test, lda_pred
    )
roc_auc_log_lda = metrics.auc(false_positive_rate_lda, true_positive_rate_lda)
plot_roc(roc_auc_log_lda, false_positive_rate_lda, true_positive_rate_lda)
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
shrinkings = [True, False]
prob = [True, False]
parameters = dict(
    kernel=kernels, shrinking=shrinkings, probability=prob
    )
svc = GridSearchCV(svm.SVC(), parameters, cv=5, scoring='accuracy')
svc.fit(X, y.ravel())
svc_opt = svc.best_estimator_
print(svc.best_params_)
print(svc.best_score_)
svc = svm.SVC(kernel='linear', probability=True)
svc.fit(X_train, y_train.ravel())
svc_pred = svc.predict(X_test)
metrics.accuracy_score(svc_pred, y_test)
confusion_matrix = metrics.confusion_matrix(y_test, svc_pred)
confusion_matrix
false_positive_rate_svm, true_positive_rate_svm, thresholds = metrics.roc_curve(
    y_test, svc_pred
    )
roc_auc_log_svm = metrics.auc(false_positive_rate_svm, true_positive_rate_svm)
plot_roc(roc_auc_log_svm, false_positive_rate_svm, true_positive_rate_svm)
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
neighbors = range(5, 16, 2)
parameters=dict(algorithm=algorithm, n_neighbors=neighbors)
knn = GridSearchCV(
    KNeighborsClassifier(), parameters, cv=5,scoring='accuracy')
knn.fit(X, y.ravel())
knn_opt = knn.best_estimator_
print(knn.best_params_)
print(knn.best_score_)
knn = KNeighborsClassifier(algorithm = 'auto', n_neighbors=11)
knn.fit(X_train, y_train.ravel())
knn_pred = knn.predict(X_test)
score = metrics.accuracy_score(knn_pred, y_test)
confusion_matrix = metrics.confusion_matrix(y_test, knn_pred)
print(score)
print(confusion_matrix)
false_positive_rate_knn, true_positive_rate_knn, thresholds = metrics.roc_curve(
    y_test, knn_pred
    )
roc_auc_log_knn = metrics.auc(false_positive_rate_knn, true_positive_rate_knn)
plot_roc(roc_auc_log_knn, false_positive_rate_knn, true_positive_rate_knn)
criterions = ['gini', 'entropy']
parameters = dict(criterion=criterions)
dtc = GridSearchCV(
    DecisionTreeClassifier(), parameters, cv=5, scoring='accuracy'
)
dtc.fit(X, y.ravel())
dtc_opt = dtc.best_estimator_
print(dtc.best_params_)
print(dtc.best_score_)
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train.ravel())
dtc_pred = dtc.predict(X_test)
score = metrics.accuracy_score(dtc_pred, y_test)
confusion_matrix = metrics.confusion_matrix(y_test, dtc_pred)
print(score)
print(confusion_matrix)
false_positive_rate_dtc, true_positive_rate_dtc, thresholds = metrics.roc_curve(
    y_test, dtc_pred
    )
roc_auc_log_dtc = metrics.auc(false_positive_rate_dtc, true_positive_rate_dtc)
plot_roc(roc_auc_log_dtc, false_positive_rate_dtc, true_positive_rate_dtc)
bagging = BaggingClassifier(n_estimators=500)
bagging.fit(X_train, y_train.ravel())
bagging_pred = bagging.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, bagging_pred)
score = metrics.accuracy_score(bagging_pred, y_test)
print(score)
print(confusion_matrix)
false_positive_rate_bagging, true_positive_rate_bagging, thresholds = metrics.roc_curve(
    y_test, bagging_pred
    )
roc_auc_log_bagging = metrics.auc(false_positive_rate_bagging, true_positive_rate_bagging)
plot_roc(roc_auc_log_bagging, false_positive_rate_bagging, true_positive_rate_bagging)
parameters = {
    'n_estimators': [10, 100, 250, 500]
}
rfc = GridSearchCV(
    RandomForestClassifier(), parameters, cv=5, scoring='accuracy'
)
rfc.fit(X, y.ravel())
rfc_opt = rfc.best_estimator_
print(rfc.best_params_)
print(rfc.best_score_)
rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train, y_train.ravel())
rfc_pred = rfc.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, rfc_pred)
score = metrics.accuracy_score(rfc_pred, y_test)
print(score)
print(confusion_matrix)
false_positive_rate_rfc, true_positive_rate_rfc, thresholds = metrics.roc_curve(
    y_test, rfc_pred
    )
roc_auc_log_rfc = metrics.auc(false_positive_rate_rfc, true_positive_rate_rfc)
plot_roc(roc_auc_log_rfc, false_positive_rate_rfc, true_positive_rate_rfc)
parameters = {
    'n_estimators': [10, 100, 250, 500],
    'loss': ['deviance', 'exponential'],
    'criterion': ['friedman_mse', 'mse', 'mae'],
    'max_depth': np.arange(3, 10)
}
boosting = GridSearchCV(
    GradientBoostingClassifier(), parameters, cv=5, scoring='accuracy'
)
boosting.fit(X, y.ravel())
boosting_opt = boosting.best_estimator_
print(boosting.best_params_)
print(boosting.best_score_)
gbc = GradientBoostingClassifier(
    n_estimators=500, criterion='mse', loss='exponential'
    )
gbc.fit(X_train, y_train.ravel())
gbc_pred = gbc.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, gbc_pred)
score = metrics.accuracy_score(gbc_pred, y_test)
print(score)
print(confusion_matrix)
false_positive_rate_gbc, true_positive_rate_gbc, thresholds = metrics.roc_curve(
    y_test, gbc_pred
    )
roc_auc_log_gbc = metrics.auc(false_positive_rate_gbc, true_positive_rate_gbc)
plot_roc(roc_auc_log_gbc, false_positive_rate_gbc, true_positive_rate_gbc)