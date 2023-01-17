import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
voice = pd.read_csv('../input/voicegender/voice.csv')
voice.info()
voice.head()
voice.describe()
voice.isnull().sum()
voice.isnull().any(axis=1).sum()
sns.countplot(voice['label'])
voice.groupby('label').mean()
def cohens_d(feature):

    m1 = voice[voice['label']=='male'][feature].mean()

    m2 = voice[voice['label']=='female'][feature].mean()

    n1 = voice[voice['label']=='male'][feature].size

    n2 = voice[voice['label']=='female'][feature].size

    s1 = voice[voice['label']=='male'][feature].std()

    s2 = voice[voice['label']=='female'][feature].std()

    s = np.sqrt((((n1-1)*s1**2) + ((n2-1)*s2**2)) / (n1+n2-2))

    d = (m1 - m2) / s

    return np.abs(d)
cohens_d_effect = pd.Series([cohens_d(i) for i in voice.columns[:-1]], index= voice.columns[:-1])
plt.figure(figsize=(20,5))

plt.plot(cohens_d_effect)
voice.groupby('label').mean().loc['female']/voice.groupby('label').mean().loc['male']
plt.figure(figsize=(20,5))

plt.plot((voice.groupby('label').mean().loc['female']/voice.groupby('label').mean().loc['male']))

plt.plot([1]*20, '--')

plt.title('Ratio of female to male values')

plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(voice.corr(), cmap='Spectral', annot=True)

plt.show()
voice.columns
long_voice = pd.melt(voice, id_vars='label', value_vars=voice.columns[:-1], var_name='properties')
g = sns.FacetGrid(long_voice, col='properties', col_wrap=5, hue='label', sharex=False, sharey=False, height=4)

g = g.map(sns.kdeplot, 'value').add_legend().set_titles("{col_name}").set_axis_labels('')
df1 = voice.iloc[:, :10]

df1 =  (df1-df1.mean())/df1.std()
plt.figure(figsize=(20,6))

sns.violinplot(data=df1)
df1 = pd.concat([df1, voice['label']], axis=1)
df1 = pd.melt(df1, id_vars='label', var_name='properties')
plt.figure(figsize=(20,7))

sns.violinplot(x='properties', y='value', hue='label', split=True, inner='quart', data=df1)
df2 = voice.iloc[:, 10:-1]

df2 =  (df2-df2.mean())/df2.std()
plt.figure(figsize=(20,6))

sns.violinplot(data=df2)
df2 = pd.melt(pd.concat([df2, voice['label']], axis=1), id_vars='label', var_name='properties')
plt.figure(figsize=(20,7))

sns.violinplot(x='properties', y='value', hue='label', split=True, inner='quart', data=df2)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, precision_recall_curve
X = voice.iloc[:, :-1]

y = voice['label']
def preprocess(X, y, rand):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = rand)

    min_ = X_train.min()

    max_ = X_train.max()

    X_train = (X_train - min_)/(max_ - min_)

    X_test = (X_test - min_)/(max_ - min_)

    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = preprocess(X, y, 53)
def scores(X, y, clf, n):

    """

        X, y are input and output variables.

        clf is classifier algorithm

        n is number of random states used for splitting the dataframe

        this function returns array of scores for random states 0 to n.

    """

    scores = []

    for i in range(n):

        X_train, X_test, y_train, y_test = preprocess(X, y, i)

        clf.fit(X_train, y_train)

        scores.append(clf.score(X_test, y_test))

    return np.array(scores)
from sklearn.ensemble import ExtraTreesClassifier
score_list = scores(X, y, ExtraTreesClassifier(n_estimators=200), 100)
plt.figure(figsize=(15,5))

plt.plot(score_list)

plt.xlabel('random state')

plt.ylabel('mean accuracy score')
score_list.mean()
cross_val_score(ExtraTreesClassifier(criterion='entropy', n_estimators=200), X_train, y_train, cv=5).mean()
etc = ExtraTreesClassifier(n_estimators=200, criterion='entropy')
etc.fit(X_train, y_train)
predictions = etc.predict(X_test)
accuracy_score(y_test, predictions)
print(classification_report(y_test, predictions))
fig, axes = plt.subplots(1,1, figsize=(10,5))

sns.heatmap(confusion_matrix(y_test, predictions), annot=True, ax=axes)
pred_prob = etc.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:, 1], pos_label='male')
precision, recall, thresholds = precision_recall_curve(y_test, pred_prob[:, 1], pos_label='male')
fig, axes = plt.subplots(1,2, figsize=(15,5))

axes[0].plot(fpr, tpr)

axes[0].plot([0, 1], [0, 1],'r--')

axes[0].set_xlim([-0.05, 1.0])

axes[0].set_ylim([0.0, 1.05])

axes[0].set_xlabel('False Positive Rate')

axes[0].set_ylabel('True Positive Rate')



axes[1].plot(recall, precision)

axes[1].plot([0, 1], [0, 1],'r--')

axes[1].set_xlim([0.0, 1.05])

axes[1].set_ylim([0.0, 1.05])

axes[1].set_xlabel('Recall')

axes[1].set_ylabel('Precision')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5), gridspec_kw={'width_ratios': [1, 2]})

sns.heatmap(pred_prob[np.argsort(pred_prob[:, 0])], ax=ax1)

ax2.plot(pred_prob[np.argsort(pred_prob[:, 0])])

ax2.set_xlabel('test case number')

ax2.set_ylabel('probability')

ax2.legend(['female', 'male'])
feature_imp = pd.DataFrame(etc.feature_importances_, voice.iloc[:, :-1].columns, columns=['importance']).sort_values(by='importance', ascending=False)
feature_imp.head()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
logreg = LogisticRegression(solver='lbfgs', max_iter=500)
cross_val_score(logreg, X_train, y_train, cv=5).mean()
logreg.fit(X_train, y_train)
logreg.score(X_test, y_test)
parameters = {'solver':( 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'), 'C':[0.001,0.01,.1,1,5,10,25,100]}
clf = GridSearchCV(logreg, parameters)
clf.fit(X_train, y_train)
clf.best_params_
clf.best_score_
parameters = {'solver':( 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'), 'C':[i for i in range(1,100)]}
clf = RandomizedSearchCV(logreg, param_distributions=parameters, n_iter=50)
clf.fit(X_train, y_train)
clf.best_params_
clf.best_score_
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
accuracy_score(y_test, predictions)
cross_val_score(rfc, X_train, y_train, cv=5).mean()
print(classification_report(y_test, predictions))
fig, axes = plt.subplots(1,1, figsize=(10,5))

sns.heatmap(confusion_matrix(y_test, predictions), annot=True, ax=axes)
pred_prob = rfc.predict_proba(X_test)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5), gridspec_kw={'width_ratios': [1, 2]})

sns.heatmap(pred_prob[np.argsort(pred_prob[:, 0])], ax=ax1)

ax2.plot(pred_prob[np.argsort(pred_prob[:, 0])])

ax2.set_xlabel('test case number')

ax2.set_ylabel('probability')

ax2.legend(['female', 'male'])
feature_imp = pd.DataFrame(rfc.feature_importances_, voice.iloc[:, :-1].columns, columns=['importance']).sort_values(by='importance', ascending=False)
feature_imp.head()
from sklearn.svm import SVC
svc = SVC(C=10, gamma='scale')
svc.fit(X_train, y_train)
predictions = svc.predict(X_test)
accuracy_score(y_test, predictions)
cross_val_score(svc, X_train, y_train, cv=5).mean()
pred_prob = svc.decision_function(X_test)
from sklearn.calibration import calibration_curve
fop, mpv = calibration_curve(y_test, pred_prob, normalize=True, n_bins=10)
plt.plot(mpv, fop, '*-')

plt.plot([0,1])
from sklearn.calibration import CalibratedClassifierCV
ccc = CalibratedClassifierCV(svc, 'sigmoid')
ccc.fit(X_train,  y_train)
pred_prob = ccc.predict_proba(X_test)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5), gridspec_kw={'width_ratios': [1, 2]})

sns.heatmap(pred_prob[np.argsort(pred_prob[:, 0])], ax=ax1)

ax2.plot(pred_prob[np.argsort(pred_prob[:, 0])])

ax2.set_xlabel('test case number')

ax2.set_ylabel('probability')

ax2.legend(['female', 'male'])
fop1, mpv1 = calibration_curve(y_test, ccc.predict_proba(X_test)[:, 1], n_bins=10, normalize=True)
plt.plot(mpv1, fop1, '^-', label='calibrated')

plt.plot(mpv, fop, '*-')

plt.plot([0,1])

plt.legend()
ccc.score(X_test, y_test)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
cross_val_score(gbc, X_train, y_train, cv=5).mean()
gbc.fit(X_train, y_train)
predictions = gbc.predict(X_test)
accuracy_score(y_test, predictions)
print(classification_report(y_test, predictions))
fig, axes = plt.subplots(1,1, figsize=(10,5))

sns.heatmap(confusion_matrix(y_test, predictions), annot=True, ax=axes)