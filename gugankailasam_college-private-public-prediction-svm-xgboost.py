import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

plt.style.use("seaborn-dark")
np.random.seed(42)
data = pd.read_csv('../input/ISLR-Auto/College.csv').drop('Unnamed: 0', axis=1)
print(data.shape)
data.head()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')

pie_data = data['Private']

labels = pie_data.unique()
target = pie_data.value_counts()
ax.pie(target, labels = labels,autopct='%1.2f%%')
plt.show()
data.isna().values.any() # null value
col_kde = data.columns.drop('Private')

fig, axes = plt.subplots(nrows=len(col_kde), ncols=2, figsize=(10,60))
fig.subplots_adjust(hspace=0.20)
axes = axes.flatten()
count = 0

for i, col in enumerate(col_kde):
    sns.kdeplot(data.loc[data['Private'] == 'Yes', col], Label='Private', ax=axes[count])
    sns.kdeplot(data.loc[data['Private'] == 'No', col], Label='Public', ax=axes[count])
    axes[count].set(title=col)
    count += 1
    sns.boxplot(x='Private', y=col, data=data, ax=axes[count])
#     axes[count].set(title=col)
    count += 1
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Init
data_vif = data.drop(['Private', 'Terminal', 'Top25perc', 'Enroll', 
                      'PhD', 'Room.Board', 'Accept', 'Grad.Rate', 'Outstate', 'Books', 'Expend', 'F.Undergrad'], axis=1)

#VIF calculation
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(data_vif.values, i) for i in range(data_vif.shape[1])]
vif['Features'] = data_vif.columns
vif.sort_values(by='VIF', ascending=False)
# Dropping certain features

features_to_drop = ['Top25perc', 'Books', 'Private']

X = data.loc[:, data.columns.drop(features_to_drop)].copy()
y = data.loc[:, 'Private'].copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y)
y = le.transform(y)
print(le.classes_)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y,
                                                   stratify = y,
                                                   test_size = 0.20)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
model_for_cv = clf

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='f1')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model_for_cv = model

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='f1')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
clf = SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 20, 30], 'gamma':['auto', 'scale']}
clf = GridSearchCV(clf, parameters, scoring='f1')
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8]}
clf = GridSearchCV(neigh, parameters, scoring='f1')
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)
from xgboost import XGBClassifier
model = XGBClassifier()
weights = [1, 72, 99, 100, 110]
param_grid = dict(scale_pos_weight=weights)
clf = GridSearchCV(model, param_grid, scoring='f1')
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)

grid_result = clf

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
features_to_drop = ['Private']

X = data.loc[:, data.columns.drop(features_to_drop)].copy()
y = data.loc[:, 'Private'].copy()
X.shape
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y)
y = le.transform(y)
print(le.classes_)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y,
                                                   stratify = y,
                                                   test_size = 0.20)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
model_for_cv = clf

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='f1')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model_for_cv = model

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='f1')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
clf = SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 20, 30], 'gamma':['auto', 'scale']}
clf = GridSearchCV(clf, parameters, scoring='f1')
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8]}
clf = GridSearchCV(neigh, parameters, scoring='f1')
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)
from xgboost import XGBClassifier
model = XGBClassifier()
weights = [1, 72, 99, 100, 110]
param_grid = dict(scale_pos_weight=weights)
clf = GridSearchCV(model, param_grid, scoring='f1')
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)

grid_result = clf

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
features_to_drop = ['Private', 'Terminal', 'Top25perc', 'Enroll', 
                      'PhD', 'Room.Board', 'Accept', 'Grad.Rate', 'Outstate', 'Books']

X = data.loc[:, data.columns.drop(features_to_drop)].copy()
y = data.loc[:, 'Private'].copy()
X.shape
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y)
y = le.transform(y)
print(le.classes_)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y,
                                                   stratify = y,
                                                   test_size = 0.20)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
model_for_cv = clf

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='f1')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model_for_cv = model

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='f1')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
clf = SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 20, 30], 'gamma':['auto', 'scale']}
clf = GridSearchCV(clf, parameters, scoring='f1')
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8]}
clf = GridSearchCV(neigh, parameters, scoring='f1')
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)
from xgboost import XGBClassifier
model = XGBClassifier()
weights = [1, 72, 99, 100, 110]
param_grid = dict(scale_pos_weight=weights)
clf = GridSearchCV(model, param_grid, scoring='f1')
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_score_)

grid_result = clf

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
clf = SVC(C=1, gamma='auto', kernel='linear', probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("----Classification Report----")
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, label='Classification (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()