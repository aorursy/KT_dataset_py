import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import scipy.stats as stats



from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from sklearn.model_selection import train_test_split



from sklearn.metrics import roc_auc_score, roc_curve
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.set_index('ID_code', inplace=True)

test.set_index('ID_code', inplace=True)
ALPHA = 0.0001
corrs = pd.Series(index=train.columns)

p_values = pd.Series(index=train.columns)

for var in train.columns:

    if var != 'target':

        res = stats.pearsonr(train['target'], train[var])

        corrs[var] = res[0]

        p_values[var] = res[1]

corr_df = pd.DataFrame({'corrs':corrs,'p_values':p_values})



fig, ax = plt.subplots()

ax.plot(corr_df.sort_values(by='corrs')['corrs'])

ax2 = plt.twinx(ax=ax)

ax2.plot(corr_df.sort_values(by='corrs')['p_values'])

ax.set_ylabel('correlation'); ax2.set_ylabel('p-value')

plt.title('correlation with target and p-value by variable');
sigvars = corr_df.loc[corr_df.p_values < ALPHA]

print(len(sigvars))



fig, ax = plt.subplots()

ax.plot(sigvars.sort_values(by='corrs')['corrs'])

ax2 = plt.twinx(ax=ax)

ax2.plot(sigvars.sort_values(by='corrs')['p_values'])

ax.set_ylabel('r'); ax2.set_ylabel('p')

plt.title('correlation with target and p-value by variable\n(sigvars)');
# chartdata = train.copy()[sigvars.sample(10).index.tolist() + ['target']].sample(10000)

# sns.pairplot(chartdata, hue='target')
train = pd.concat([train, train.loc[train.target == 1]])

train = pd.concat([train, train.loc[train.target == 1]])

train = pd.concat([train, train.loc[train.target == 1]])

sns.countplot(train.target); plt.title('Imbalanced Classes -- fixed?');
y = train.pop('target')

X = train[sigvars.index.tolist()] # feature selection

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = LogisticRegressionCV(scoring='roc_auc', n_jobs=2, random_state=42)

clf.fit(X_train,y_train)
pred_train = pd.DataFrame(clf.predict_proba(X_train))[1]

pred_test = pd.DataFrame(clf.predict_proba(X_test))[1]



print('train:',roc_auc_score(y_train, pred_train))

print('test:',roc_auc_score(y_test, pred_test))
sns.distplot(pred_train)
curve = roc_curve(y_train, pred_train)

plt.plot(curve[0], curve[1])

plt.title('ROC Curve (Train)');
clf.fit(X, y)

preds = pd.DataFrame(clf.predict_proba(test[sigvars.index.tolist()]), index=test.index)[1]

preds.rename('target', inplace=True)

preds.to_csv('submission.csv', header=True)