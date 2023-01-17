# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.shape
data.describe()
data.Time.hist(bins=48)
plt.show()
data['Hour'] = data['Time'].map(lambda x : x // 3600 % 24)
data['Day'] = data['Time'].map(lambda x: x // 3600 // 24 + 1)
data.head()
data.Hour.hist(bins=24)
plt.show()
data.Amount.value_counts().head(10)
data.Amount.hist(bins=50)
plt.show()
data.corr().Class.sort_values()
data_fraud = data[data.Class==1]
data_not = data[data.Class==0]
data_fraud.describe()
data_fraud.Hour.hist(bins=24)
plt.show()
data_fraud.Amount.value_counts().head(10)
data_fraud.Amount.sort_values(ascending=False).head(10)
data_fraud.Amount.hist(bins=22)
plt.show()
data_fraud.describe() - data.describe()
(data_fraud.describe() - data.describe()).loc['mean'].sort_values()
(data_fraud.describe() - data.describe()).loc['std'].sort_values()
X = data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Hour']]
y = data['Class']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

log_clf = LogisticRegression()

param_grid = [{
    'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    'class_weight': ['balanced', None]
}]
grid_search = GridSearchCV(log_clf, param_grid, cv=3, scoring='roc_auc')
grid_search.fit(X_scaled, y)
grid_search.best_params_
grid_search.best_score_
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

log_clf = LogisticRegression()
for i in range(1, 11, 1):
    log_clf.C = i/10000
    y_score = cross_val_predict(log_clf, X_scaled, y, method='decision_function', cv=5, n_jobs=-1)
    print('C={:.4f}'.format(log_clf.C), roc_auc_score(y, y_score))
log_clf = LogisticRegression()
for i in range(1, 11, 1):
    log_clf.C = i/10000 + 0.001
    y_score = cross_val_predict(log_clf, X_scaled, y, cv=5, method='decision_function', n_jobs=-1)
    print('C={:.4f}'.format(log_clf.C), roc_auc_score(y, y_score))
log_clf = LogisticRegression(C=0.0017)
y_scores = cross_val_predict(log_clf, X_scaled, y, cv=5, method='decision_function', n_jobs=-1)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.show()
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier()
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
y_probas_rnd = cross_val_predict(rnd_clf, X_scaled, y, cv=3, method='predict_proba', n_jobs=-1)
y_scores_rnd = y_probas_rnd[:, 1]
fpr_rnd, tpr_rnd, thresholds_rnd = roc_curve(y, y_scores_rnd)
roc_auc_score(y, y_scores_rnd)
rnd_clf.fit(X_scaled, y)
for name, score in zip(X.columns, rnd_clf.feature_importances_):
    print(name, score)
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d', title='Data')
ax1.scatter(X.V17, X.V12, X.V14, c=y)
ax1.set_xlabel('V17')
ax1.set_ylabel('V12')
ax1.set_zlabel('V14')


ax2 = fig.add_subplot(122, projection='3d', title='Fraud Data Only')
ax2.scatter(X.V17[y==1], X.V12[y==1], X.V14[y==1], c='y')
ax2.set_xlabel('V17')
ax2.set_ylabel('V12')
ax2.set_zlabel('V14')

plt.show()
plt.plot(X.V17[y==1], X.V14[y==1], 'y.')
plt.plot([5, -5], [-9, 2])
plt.xlim([-30, 10])
plt.ylim([-20, 5])
plt.show()
from sklearn.ensemble import IsolationForest
iso_clf = IsolationForest(n_estimators=500, random_state=42)
iso_clf.fit(X_scaled)
y_pred = iso_clf.predict(X_scaled)
y_pred.sum()
from sklearn.svm import SVC
svm_clf = SVC()
y_score = cross_val_predict(svm_clf, X_scaled, y, 
                            method='decision_function', cv=5, n_jobs=-1)
roc_auc_score(y, y_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(), n_estimators=400,
    algorithm='SAMME.R', learning_rate=0.6)
y_score = cross_val_predict(ada_clf, X_scaled, y, 
                            method='decision_function', cv=5, n_jobs=-1)
roc_auc_score(y, y_score)
import xgboost
xgb_clf = xgboost.XGBClassifier()
xgb_clf.fit(X_scaled, y)
y_xgb_pred = xgb_clf.predict(X_scaled)
from sklearn.model_selection import cross_val_score
y_score = cross_val_score(xgb_clf, X_scaled, y, 
                            scoring='roc_auc', cv=5, n_jobs=-1)
y_score
y_score.mean()
xgb_clf = xgboost.XGBClassifier(tree_method = 'hist')
from sklearn.model_selection import cross_val_score
y_score = cross_val_score(xgb_clf, X_scaled, y, 
                            scoring='roc_auc', cv=5, n_jobs=-1)
y_score
y_score.mean()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2D = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.plot(X2D[:, 0][y==0], X2D[:, 1][y==0], 'b.')
plt.plot(X2D[:, 0][y==1], X2D[:, 1][y==1], 'r.')
plt.show()
from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X_scaled)
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d', title='Clustered Data')
ax1.scatter(X.V17, X.V12, X.V14, c=y_pred, cmap='rainbow')
ax1.set_xlabel('V17')
ax1.set_ylabel('V12')
ax1.set_zlabel('V14')

ax2 = fig.add_subplot(122, projection='3d', title='Data')
ax2.scatter(X.V17, X.V12, X.V14, c=y)
ax2.set_xlabel('V17')
ax2.set_ylabel('V12')
ax2.set_zlabel('V14')

plt.show()
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X_scaled)
FPR = []
TPR = []
real_Negative = 284807 - 492
real_Positive = 492
densities = gm.score_samples(X_scaled)
for i in range(10000):
    threshold = (i + 1) / 100
    density_threshold = np.percentile(densities, threshold)
    anomalies = data[densities < density_threshold]
    TP = anomalies.Class.sum()
    FP = anomalies.Class.count() - TP
    FPR.append(FP / real_Negative)
    TPR.append(TP / real_Positive)
plot_roc_curve(FPR, TPR)
score = 0
for i in range(10000-1):
    score += (TPR[i+1] + TPR[i]) * (FPR[i+1] - FPR[i]) / 2
score
from sklearn.mixture import BayesianGaussianMixture
bgm = BayesianGaussianMixture(n_components=3, n_init=10)
bgm.fit(X_scaled)
FPR = []
TPR = []
real_Negative = 284807 - 492
real_Positive = 492
densities = bgm.score_samples(X_scaled)
for i in range(10000):
    threshold = (i + 1) / 100
    density_threshold = np.percentile(densities, threshold)
    anomalies = data[densities < density_threshold]
    TP = anomalies.Class.sum()
    FP = anomalies.Class.count() - TP
    FPR.append(FP / real_Negative)
    TPR.append(TP / real_Positive)
plot_roc_curve(FPR, TPR)
score = 0
for i in range(10000-1):
    score += (TPR[i+1] + TPR[i]) * (FPR[i+1] - FPR[i]) / 2
score
