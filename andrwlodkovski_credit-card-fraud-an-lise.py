# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import auc, precision_recall_curve, recall_score
from xgboost import XGBClassifier
from scipy import stats
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

X = df.drop("Class", axis=1)
y = df['Class']

df.head(10)
df.describe()
df['Class'].hist()
df['Class'].value_counts(normalize=True)
plt.rcParams['figure.figsize'] = (20, 20)
plt.rcParams['font.size'] = 10

sns.heatmap(df.corr(), annot=True, fmt='.2f')
sns.pairplot(data=df.sample(5000), hue='Class', palette='husl')
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X,y)
importance_dict = dict(zip(X.columns,forest.feature_importances_))
importance_dict = {k: v for k, v in reversed(sorted(importance_dict.items(), key=lambda item: item[1]))}
importance_dict
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['font.size'] = 10
def calc_vif(tabela):
  vif = pd.DataFrame()
  vif['variaveis'] = tabela.columns
  vif['vif'] = [variance_inflation_factor(tabela.values, i) for i in range(tabela.shape[1])]

  return vif
vif = calc_vif(df)
vif.sort_values(by=['vif'], ascending=False)
sns.distplot(df[df['Class'] == 0]['V3'])
sns.distplot(df[df['Class'] == 1]['V3'])
sns.distplot(df[df['Class'] == 0]['V4'])
sns.distplot(df[df['Class'] == 1]['V4'])
sns.distplot(df[df['Class'] == 0]['V11'])
sns.distplot(df[df['Class'] == 1]['V11'])
sns.distplot(df[df['Class'] == 0]['V12'])
sns.distplot(df[df['Class'] == 1]['V12'])
sns.distplot(df[df['Class'] == 0]['V14'])
sns.distplot(df[df['Class'] == 1]['V14'])
#sort para priorizar a classe 1
sns.scatterplot(data=df.sort_values(by=['Class'], ascending=True), x='Amount', y='V2', hue='Class')
sns.scatterplot(data=df.sort_values(by=['Class'], ascending=True), x='Amount', y='V5', hue='Class')
sns.scatterplot(data=df.sort_values(by=['Class'], ascending=True), x='Amount', y='V7', hue='Class')
sns.scatterplot(data=df.sort_values(by=['Class'], ascending=True), x='Amount', y='V20', hue='Class')
#usando IQR para detectar os outliers de cada classe
outliers = []
columns = df.columns[:-1]

for column in columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    outliers.append(df[(df[column] > q3 + iqr * 1.5) | (df[column] < q1 - iqr * 1.5)])

outliers_quantities = [col.count().max() for col in outliers]
outliers_dict = dict(zip(columns, outliers_quantities))
outliers_dict = {k: v for k, v in reversed(sorted(outliers_dict.items(), key=lambda item: item[1]))}
outliers_dict
sns.boxplot(data=df, y='V27', x='Class')
g = sns.boxplot(x=df['V27'])
g.set_xlim(left=-5, right=5)
sns.boxplot(data=df, y='Amount', x='Class')
g = sns.boxplot(x=df['Amount'])
g.set_xlim(left=-100, right=1000)
sns.boxplot(data=df, y='V28', x='Class')
g = sns.boxplot(x=df['V28'])
g.set_xlim(left=-5, right=5)
sns.boxplot(data=df, y='V11', x='Class')
sns.boxplot(data=df, y='V22', x='Class')
sns.boxplot(data=df, y='V15', x='Class')
preds_auc = {}
preds_ks = {}
preds_rec = {}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
preds = forest.predict(X_test)
probs = forest.predict_proba(X_test)
probs_true = [x[1] for x in probs]
probs_false = [x[0] for x in probs]

model_name = "Random Forest"
precision, recall, _ = precision_recall_curve(y_test, probs_true)
a = auc(recall, precision)
ks = stats.ks_2samp(probs_true,probs_false)
rec = recall_score(y_test, preds)
preds_auc[model_name] = a
preds_ks[model_name] = ks.statistic
preds_rec[model_name] = rec
print("AUC-PR:",a)
print("KS statistic:", ks.statistic,"pvalue:",ks.pvalue)
print("Recall:",rec)
sns.heatmap(confusion_matrix(y_test, preds),annot=True)
g = sns.distplot(probs_false, kde=False)
g = sns.distplot(probs_true, kde=False)
xgb = XGBClassifier(n_estimators=100, learning_rate=0.5)
xgb.fit(X_train, y_train)
preds = xgb.predict(X_test)
probs = xgb.predict_proba(X_test)
probs_true = [x[1] for x in probs]
probs_false = [x[0] for x in probs]

model_name = "XGB Classifier"
precision, recall, _ = precision_recall_curve(y_test, probs_true)
a = auc(recall, precision)
ks = stats.ks_2samp(probs_true,probs_false)
rec = recall_score(y_test, preds)
preds_auc[model_name] = a
preds_ks[model_name] = ks.statistic
preds_rec[model_name] = rec
print("AUC-PR:",a)
print("KS statistic:", ks.statistic,"pvalue:",ks.pvalue)
print("Recall:",rec)
sns.heatmap(confusion_matrix(y_test, preds),annot=True)
lr = LogisticRegression(random_state=0, max_iter=1000000)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)
probs = lr.predict_proba(X_test)
probs_true = [x[1] for x in probs]
probs_false = [x[0] for x in probs]

model_name = "Logistic Regression"
precision, recall, _ = precision_recall_curve(y_test, probs_true)
a = auc(recall, precision)
ks = stats.ks_2samp(probs_true,probs_false)
rec = recall_score(y_test, preds)
preds_auc[model_name] = a
preds_ks[model_name] = ks.statistic
preds_rec[model_name] = rec
print("AUC-PR:",a)
print("KS statistic:", ks.statistic,"pvalue:",ks.pvalue)
print("Recall:",rec)
sns.heatmap(confusion_matrix(y_test, preds),annot=True)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
probs = knn.predict_proba(X_test)
probs_true = [x[1] for x in probs]
probs_false = [x[0] for x in probs]

model_name = "K Neighbors Classifier"
precision, recall, _ = precision_recall_curve(y_test, probs_true)
a = auc(recall, precision)
ks = stats.ks_2samp(probs_true,probs_false)
rec = recall_score(y_test, preds)
preds_auc[model_name] = a
preds_ks[model_name] = ks.statistic
preds_rec[model_name] = rec
print("AUC-PR:",a)
print("KS statistic:", ks.statistic,"pvalue:",ks.pvalue)
print("Recall:",rec)
sns.heatmap(confusion_matrix(y_test, preds),annot=True)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
preds = tree.predict(X_test)
probs = tree.predict_proba(X_test)
probs_true = [x[1] for x in probs]
probs_false = [x[0] for x in probs]

model_name = "Decision Tree Classifier"
precision, recall, _ = precision_recall_curve(y_test, probs_true)
a = auc(recall, precision)
ks = stats.ks_2samp(probs_true,probs_false)
rec = recall_score(y_test, preds)
preds_auc[model_name] = a
preds_ks[model_name] = ks.statistic
preds_rec[model_name] = rec
print("AUC-PR:",a)
print("KS statistic:", ks.statistic,"pvalue:",ks.pvalue)
print("Recall:",rec)
sns.heatmap(confusion_matrix(y_test, preds),annot=True)
def sort_dict(dictionary):
    return {k: v for k, v in reversed(sorted(dictionary.items(), key=lambda item: item[1]))}
sort_dict(preds_auc)
sort_dict(preds_ks)
sort_dict(preds_rec)
sort_dict(dict(zip(X.columns, forest.feature_importances_)))
sort_dict(dict(zip(X.columns, xgb.feature_importances_)))
sort_dict(dict(zip(X.columns, tree.feature_importances_)))
