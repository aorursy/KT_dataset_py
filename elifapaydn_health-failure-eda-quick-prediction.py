# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation: CROSS VALIDATION

from sklearn.model_selection import cross_val_predict #prediction



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import plot_confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, classification_report





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

data.head()
data.shape
data.isnull().sum()
data.describe()
data.info()
cat_cols=[c for c in data.columns if data[c].isin([0,1]).all()]

cat_cols.remove('DEATH_EVENT')

print(cat_cols)
num_cols=[c for c in data.columns if c not in cat_cols]

num_cols.remove('DEATH_EVENT')

num_cols
c=pd.melt(data, id_vars='DEATH_EVENT', value_vars=cat_cols)

grid=sns.FacetGrid(c, col_wrap=3, col='variable', sharex=False, sharey=True, height=4)

grid.map(sns.barplot, 'value', 'DEATH_EVENT', palette='Set2', order=[0,1])
c=pd.melt(data, id_vars='DEATH_EVENT', value_vars=cat_cols)

sns.catplot(x='value', hue='DEATH_EVENT', col='variable', data=c, palette='RdYlBu', kind="count", 

            col_wrap=3, height=4, aspect=.7, sharex=False, sharey=True)

data[num_cols]
melt=pd.melt(data, value_vars=num_cols)

grid=sns.FacetGrid(melt, col='variable',col_wrap=3, sharex=False, sharey=False, height=4, aspect=1)

grid.map(sns.distplot, 'value')
melt=pd.melt(data, value_vars=num_cols)

grid=sns.FacetGrid(melt, col='variable',col_wrap=2, sharex=False, sharey=False, aspect=2)

grid.map(sns.boxplot, 'value')
sns.countplot(data['DEATH_EVENT'])
correlation=data.corr().abs()

fig=plt.figure(figsize=(15,12))

ax=plt.gca()

sns.heatmap(correlation, annot=True, cmap='RdYlBu', mask=np.triu(correlation, k=0), ax=ax)
cm = sns.light_palette("green", as_cmap=True)

columns=correlation.nlargest(5, 'DEATH_EVENT').index.tolist()

correlation.nlargest(5, 'DEATH_EVENT')[columns].style.background_gradient(cmap=cm)
X=data[columns].drop('DEATH_EVENT', axis=1)

y=data['DEATH_EVENT']

print(columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, shuffle=True)
# Feature Scaling



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
model = LogisticRegression(random_state=0)

model.fit(X_train, y_train)

log_preds = model.predict(X_test)

print('Logistic Regression ', accuracy_score(y_test, log_preds))
plot_confusion_matrix(model, X_test, y_test) 
print('Logistic Regression f1-score:', f1_score(y_test, log_preds))

print('Logistic Regression precision:', precision_score(y_test, log_preds))

print('Logistic Regression recall:', recall_score(y_test, log_preds))
print(classification_report(y_test, log_preds))
sklearn.metrics.SCORERS.keys()
#Stratified Kfold for imbalanced classification problem.

stkfold=StratifiedKFold(n_splits=5)

scores=cross_val_score(model, X, y, cv = stkfold, scoring = "f1")

scores.mean()
knn_scores=[]

for i in range(1,25):

    model=KNeighborsClassifier(n_neighbors=i)

    model.fit(X_train, y_train)

    knn_preds=model.predict(X_test)

    knn_scores.append(f1_score(y_test, knn_preds))
plt.figure(figsize=(12,8))

plt.plot(range(1,25,1), knn_scores)

ticks=plt.gca().set_xticks(range(1,25))
knn_model=KNeighborsClassifier(n_neighbors=7)

knn_model.fit(X_train, y_train)

knn_preds=model.predict(X_test)

print("f1 score:",f1_score(y_test, knn_preds))

print()

print(classification_report(y_test, knn_preds))
rf_model=RandomForestClassifier(max_leaf_nodes=100, max_depth=10,random_state=0)

rf_model.fit(X_train, y_train)

rf_preds=rf_model.predict(X_test)

print("f1 score:",f1_score(y_test, rf_preds))

print()

print(classification_report(y_test, rf_preds))