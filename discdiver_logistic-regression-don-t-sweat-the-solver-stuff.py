#!pip list # list packages 
# import packages

import numpy as np

import pandas as pd

from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

import seaborn as sns

sns.set()

%matplotlib inline
rd = load_wine()

X, y = load_wine(return_X_y=True)



df = pd.DataFrame(X, columns=rd.feature_names)

df['target'] = y

df.head()
df.target.value_counts()
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:,-1], random_state=34)
import matplotlib.pyplot as plt



mask = np.zeros_like(X_train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)]= True



plt.figure(figsize=(10,10))

plt.title("Wine Feature Correlation Matrix", fontsize=40)

x = sns.heatmap(

    X_train.corr(), 

    cmap='coolwarm',

    annot=True,

    mask=mask,

    linewidths = .5,

    vmin = -1, 

    vmax = 1,

)
# x.get_figure().savefig('images/wine_corr.png')
X_train.corr()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=34)
logistic_regression_model = LogisticRegression(random_state=34, solver='lbfgs', multi_class="auto", n_jobs=-1, C=1)

logistic_regression_model.fit(X_train, y_train)
accuracy_score = logistic_regression_model.score(X_val, y_val)

accuracy_score 
predictions = logistic_regression_model.predict(X_val)

predictions[:10]
confusion_matrix(y_val, predictions)
logistic_regression_model = LogisticRegression(random_state=34, solver='saga', multi_class="auto", n_jobs=-1, C=1)

logistic_regression_model.fit(X_train, y_train)
accuracy_score = logistic_regression_model.score(X_val, y_val)

accuracy_score 
predictions = logistic_regression_model.predict(X_val)

predictions[:10]
confusion_matrix(y_val, predictions)
from sklearn.preprocessing import MinMaxScaler

mm_scaler = MinMaxScaler()

X_train = mm_scaler.fit_transform(X_train)

X_val = mm_scaler.transform(X_val)

X_test = mm_scaler.transform(X_test)
logistic_regression_model_scaled = LogisticRegression(random_state=34, solver='saga', multi_class="auto", n_jobs=-1, C=1)

logistic_regression_model_scaled.fit(X_train, y_train)

accuracy_score = logistic_regression_model_scaled.score(X_val, y_val)

accuracy_score 
predictions = logistic_regression_model_scaled.predict(X_val)

predictions[:10]
confusion_matrix(y_val, predictions)
from sklearn.model_selection import GridSearchCV



X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:,-1], random_state=34)
solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

parameters = dict(solver=solver_list)

lr = LogisticRegression(random_state=34, multi_class="auto", n_jobs=-1, C=1)

clf = GridSearchCV(lr, parameters, cv=5)
clf.fit(X_train, y_train)
clf.cv_results_['mean_test_score']
scores = clf.cv_results_['mean_test_score']

for score, solver, in zip(scores, solver_list):

    print(f"{solver}: {score:.3f}")
sns.barplot(x=solver_list, y=scores). set_title("Wine Accuracy with Unscaled Features")
ax = sns.barplot(x=solver_list, y=scores)

ax.set_title("Wine Accuracy with Unscaled Features", fontsize = 20)
fig = ax.get_figure()

# fig.savefig('images/wine_unscaled.png')
params = dict(solver=solver_list)

log_reg = LogisticRegression(C=1, n_jobs=-1, random_state=34)

clf = GridSearchCV(log_reg, params, cv=5)

clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_test_score']



for score, solver in zip(scores, solver_list):

    print(f"  {solver} {score:.3f}" )
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

parameters = dict(solver=solver_list)

lr = LogisticRegression(random_state=34, multi_class="auto", n_jobs=-1, C=1)

clf = GridSearchCV(lr, parameters, cv=5)

clf.fit(X_train, y_train)
clf.cv_results_['mean_test_score']

scores = clf.cv_results_['mean_test_score']

for score, solver, in zip(scores, solver_list):

    print(f"{solver}: {score:.3f}")
ax =sns.barplot(x=solver_list, y=scores).set_title("Wine Accuracy with Scaled Features", fontsize="20")
fig = ax.get_figure()

# fig.savefig('images/wine_scaled.png')
from sklearn.datasets import load_breast_cancer



raw = load_breast_cancer()

X, y = load_breast_cancer(return_X_y=True)



df = pd.DataFrame(X, columns=raw.feature_names)

df['target'] = y

df.head()
df.target.value_counts()
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.loc[:,'target'], random_state=34)
params = dict(solver=solver_list)

log_reg = LogisticRegression(C=1, n_jobs=-1, random_state=34)

clf = GridSearchCV(log_reg, params, cv=5)

clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_test_score']



for score, solver in zip(scores, solver_list):

    print(f"  {solver} {score:.3f}" )
sns.barplot(x=solver_list, y=scores). set_title("Cancer Accuracy with Unscaled Features")

sns.set()
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
params = dict(solver=solver_list)

log_reg = LogisticRegression(C=1, n_jobs=-1, random_state=34)

clf = GridSearchCV(log_reg, params, cv=5)

clf.fit(X_train, y_train)
scores = clf.cv_results_['mean_test_score']



for score, solver in zip(scores, solver_list):

    print(f"  {solver} {score:.3f}" )
wine_results_scaled = zip(scores, solver_list)
sns.barplot(x=solver_list, y=scores). set_title("Cancer Accuracy with Scaled Features")
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
log_reg.intercept_
log_reg.coef_
odds = [(prob/10)/(1-(prob/10)) for prob in range(1,10)]

odds
for prob, odd in zip(range(1, 10), odds):

    print(prob/10, odd)
from math import log
log_odds = [ log(odd) for odd in odds]

log_odds
a = [1, 1, 2, 3, 4]

b = [2, 2, 3, 2, 1]

c = [4, 6, 7, 8, 9]

d = [4, 3, 4, 5, 4]



df = pd.DataFrame({'a':a,'b':b,'c':c,'d':d})
df
diagonal_value_list = []



mat = np.array(df.values)



df_cor = df.corr()

df_cor_inv = pd.DataFrame(np.linalg.inv(df.corr().values), index = df_cor.index, columns=df_cor.columns)



for a in range(len(df.columns)):

    val = df_cor_inv.iloc[a,a]

    diagonal_value_list.append(val)

    

print(dict(zip(df_cor.columns, [x.round() for x in diagonal_value_list])))