# for python 2.x and 3.x support
from __future__ import division, print_function, unicode_literals

# computation libraries used
import pandas as pd
import numpy as np

#### graphing libraries ####
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
############################


# sklearn for ML
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

# mjd to norml time conversion
from datetime import datetime, timedelta
sloan = pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
sloan.head()
sloan.info()
sloan.describe()
sloan.drop(columns=['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid', 'fiberid', 'plate'], inplace=True)
sloan.head()
f, axes = plt.subplots(2, 1, figsize=(15, 10))
sns.boxplot(y='class', x='ra', data=sloan, ax=axes[0])
sns.boxplot(y='class', x='dec', data=sloan, ax=axes[1])
f, ax = plt.subplots(figsize=(15, 10))
sns.catplot(y='redshift', x='class', data=sloan, ax=ax)
f, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
c = ['STAR', 'GALAXY', 'QSO']

for ax_id in range(3):
    sns.distplot(sloan.loc[sloan['class']==c[ax_id],'u'], hist=False, color='purple', ax=axes[ax_id], label='u')
    sns.distplot(sloan.loc[sloan['class']==c[ax_id],'g'], hist=False, color='blue', ax=axes[ax_id], label='g')
    sns.distplot(sloan.loc[sloan['class']==c[ax_id],'r'], hist=False, color='green', ax=axes[ax_id], label='r')
    sns.distplot(sloan.loc[sloan['class']==c[ax_id],'i'], hist=False, color='red', ax=axes[ax_id], label='i')
    sns.distplot(sloan.loc[sloan['class']==c[ax_id],'z'], hist=False, color='grey', ax=axes[ax_id], label='z')
    axes[ax_id].set(xlabel=c[ax_id], ylabel='Intensity')
f, axes = plt.subplots(5, 1, figsize=(16, 20))
c = ['u','g', 'r', 'i', 'z']

for idx, cls in enumerate(c):
    sns.boxplot(y='class', x=cls, data=sloan, ax=axes[idx])
# MJD starts at 17th November 1858, midnight
_MJD_BASE_TIME_ = datetime.strptime('17/11/1858 00:00', '%d/%m/%Y %H:%M')

def convertMJD(x=0):
    return _MJD_BASE_TIME_ + timedelta(days=x)
timeline_stars  = sloan.loc[sloan['class']=='STAR'  , 'mjd']
timeline_galaxy = sloan.loc[sloan['class']=='GALAXY', 'mjd']
timeline_qso    = sloan.loc[sloan['class']=='QSO'   , 'mjd']
f, ax = plt.subplots(figsize=(16, 10))
sns.distplot(timeline_stars , hist=False, label='STAR'  , ax=ax)
sns.distplot(timeline_galaxy, hist=False, label='GALAXY', ax=ax)
sns.distplot(timeline_qso   , hist=False, label='QSO'   , ax=ax)
sns.pairplot(sloan, hue='class')
sns.pairplot(sloan[['u','g','r','i','z','class']], hue='class')
f, axes = plt.subplots(1, 3, figsize=(16, 5))

star_corr = sloan.loc[sloan['class']=='STAR', ['u','g','r','i','z']].corr()
galaxy_corr = sloan.loc[sloan['class']=='GALAXY', ['u','g','r','i','z']].corr()
qso_corr = sloan.loc[sloan['class']=='QSO', ['u','g','r','i','z']].corr()

msk = np.zeros_like(star_corr)
msk[np.triu_indices_from(msk)] = True

sns.heatmap(star_corr, cmap='RdBu_r', mask=msk, ax=axes[0])
sns.heatmap(galaxy_corr, cmap='RdBu_r', mask=msk, ax=axes[1])
sns.heatmap(qso_corr, cmap='RdBu_r', mask=msk, ax=axes[2])
f, ax = plt.subplots(figsize=(16, 10))
sns.scatterplot(x='ra', y='dec', hue='class', data=sloan)
lbl = LabelEncoder()
cls_enc = lbl.fit_transform(sloan['class'])

g = go.Scatter3d(
    x=sloan['ra'], y=sloan['dec'], z=sloan['redshift'],
    mode='markers',
    marker=dict(
        color=cls_enc,
        opacity=0.5,
    )
)

g_data = [g]

layout = go.Layout(margin=dict(
    l=0, r=0, b=0, t=0
))

figure = go.Figure(data=g_data, layout=layout)

iplot(figure, filename='3d-repr-redshift')
sloan.drop(columns=['mjd'], inplace=True)
lbl_enc = LabelEncoder()
sloan['class'] = lbl_enc.fit_transform(sloan['class'])
sloan.head()
X = sloan.drop(columns=['class'])
y = sloan['class']
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=42)

for train_index, test_index in strat_split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
X_train.shape
y_train.shape
X_test.shape
y_test.shape
strat_split_val = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=42)

for train_index, val_index in strat_split.split(X_train, y_train):
    X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
X_train.shape
y_train.shape
X_val.shape
y_val.shape
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_train_scaled.head()
y_val.value_counts()
lbl_enc.classes_
knn = KNeighborsClassifier(n_jobs=-1)
knn.fit(X_train_scaled, y_train)
accuracy_score(y_val, knn.predict(X_val_scaled))
confusion_matrix(y_val, knn.predict(X_val_scaled))
log_reg = LogisticRegression(n_jobs=-1)
log_reg.fit(X_train, y_train)
accuracy_score(y_val, log_reg.predict(X_val))
confusion_matrix(y_val, log_reg.predict(X_val))
sgd_cls = SGDClassifier(n_jobs=-1)
sgd_cls.fit(X_train_scaled, y_train)
accuracy_score(y_val, sgd_cls.predict(X_val_scaled))
confusion_matrix(y_val, sgd_cls.predict(X_val_scaled))
svc_cls = SVC()
svc_cls.fit(X_train_scaled, y_train)
accuracy_score(y_val, svc_cls.predict(X_val_scaled))
confusion_matrix(y_val, svc_cls.predict(X_val_scaled))
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
accuracy_score(y_val, tree.predict(X_val))
confusion_matrix(y_val, tree.predict(X_val))
rf = RandomForestClassifier(n_jobs=-1, oob_score=True)
rf.fit(X_train, y_train)
rf.oob_score_
accuracy_score(y_val, rf.predict(X_val))
confusion_matrix(y_val, rf.predict(X_val))
xgb_cls = GradientBoostingClassifier()
xgb_cls.fit(X_train, y_train)
accuracy_score(y_val, xgb_cls.predict(X_val))
confusion_matrix(y_val, xgb_cls.predict(X_val))
etree = ExtraTreesClassifier(oob_score=True, n_jobs=-1, bootstrap=True)
etree.fit(X_train, y_train)
accuracy_score(y_val, etree.predict(X_val))
etree.oob_score_
confusion_matrix(y_val, etree.predict(X_val))
def display_scores(scores):
    print(scores)
    print('Mean: {}'.format(scores.mean()))
    print('Std: {}'.format(scores.std()))
etree_scores = cross_val_score(etree, X_train, y_train, cv=10, n_jobs=-1)
display_scores(etree_scores)
xgb_cls_scores = cross_val_score(xgb_cls, X_train, y_train, cv=10, n_jobs=-1)
display_scores(xgb_cls_scores)
tree_scores = cross_val_score(tree, X_train, y_train, cv=10, n_jobs=-1)
display_scores(tree_scores)
rf_scores = cross_val_score(rf, X_train, y_train, cv=10, n_jobs=-1)
display_scores(rf_scores)
param_grid_rf = {
    'criterion': ['gini', 'entropy'],
    'max_features': [0.5, 0.75, 0.9, 'auto'],
    'min_samples_leaf': [1, 2, 3, 4],
    'n_estimators': [5, 10, 20, 50, 75, 100]
}
cv_rf = GridSearchCV(rf, param_grid_rf, n_jobs=-1, refit=True, verbose=1)
cv_rf.fit(X_train, y_train)
cv_rf.best_params_
cv_rf.best_score_
final_model = cv_rf.best_estimator_
accuracy_score(y_test, final_model.predict(X_test))
confusion_matrix(y_test, final_model.predict(X_test))
precision_score(y_test, final_model.predict(X_test), average='weighted')
recall_score(y_test, final_model.predict(X_test), average='weighted')
f1_score(y_test, final_model.predict(X_test), average='weighted')
