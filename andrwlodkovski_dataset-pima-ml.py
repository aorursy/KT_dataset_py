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
#modelos
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
#plotting
import seaborn as sns
import matplotlib.pyplot as plt
#dados
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print(df.columns)
df.head(10)
df.describe()
sns.distplot(df['Outcome'],kde=False)
#Percentual de negativos (0) vs. positivos (1)
df['Outcome'].value_counts()/df['Outcome'].count()
sns.heatmap(df.corr(), annot=True, fmt=".2f")
sns.pairplot(df, hue="Outcome", palette="husl")
sns.swarmplot(data=df, y='Glucose', x='Outcome')
sns.scatterplot(data=df, x='Glucose', y='BMI', hue='Outcome')
sns.regplot(data=df, x='Pregnancies', y='Age')
sns.scatterplot(data=df, y='Age', x='Pregnancies', hue='Outcome')
sns.boxplot(x=df['BMI'])
bmiout = df[(df['BMI'] > 50) | (df['BMI'] < 15)]
sns.pairplot(bmiout, hue='Outcome',palette='husl', diag_kind='hist')
sns.heatmap(bmiout.corr(),annot=True, fmt='.2f')
sns.boxplot(x=df['Glucose'])
q1 = df['Glucose'].quantile(0.25)
q3 = df['Glucose'].quantile(0.75)
iqr = q3 - q1

print(q3 + 1.5 * iqr)
gluout = df[(df['Glucose'] > q3 + 1.5 * iqr) | (df['Glucose'] < q1 - 1.5 * iqr)]
sns.boxplot(x=df['Insulin'])
q1 = df['Insulin'].quantile(0.25)
q3 = df['Insulin'].quantile(0.75)
iqr = q3 - q1

print(q3 + 1.5 * iqr)
insout = df[(df['Insulin'] > q3 + 1.5 * iqr)]
sns.pairplot(insout, hue='Outcome',palette='husl', diag_kind='hist')
sns.heatmap(insout.corr(), annot=True, fmt='.2f')
sns.boxplot(x=df['SkinThickness'])
q1 = df['SkinThickness'].quantile(0.25)
q3 = df['SkinThickness'].quantile(0.75)
iqr = q3 - q1

print(q3 + 1.5 * iqr)
sknout = df[(df['SkinThickness'] > q3 + 1.5 * iqr) | (df['SkinThickness'] < q1 - 1.5 * iqr)]
sns.boxplot(x=df['Pregnancies'])
q1 = df['Pregnancies'].quantile(0.25)
q3 = df['Pregnancies'].quantile(0.75)
iqr = q3 - q1

print(q3 + 1.5 * iqr)
prgout = df[(df['Pregnancies'] > q3 + 1.5 * iqr) | (df['Pregnancies'] < q1 - 1.5 * iqr)]
sns.boxplot(x=df['BloodPressure'])
q1 = df['BloodPressure'].quantile(0.25)
q3 = df['BloodPressure'].quantile(0.75)
iqr = q3 - q1

print(q3 + 1.5 * iqr)
blpout = df[(df['BloodPressure'] > q3 + 1.5 * iqr) | (df['BloodPressure'] < q1 - 1.5 * iqr)]
sns.pairplot(blpout, hue='Outcome',palette='husl', diag_kind='hist')
sns.heatmap(blpout.corr(), annot=True, fmt='.2f')
sns.boxplot(x=df['DiabetesPedigreeFunction'])
q1 = df['DiabetesPedigreeFunction'].quantile(0.25)
q3 = df['DiabetesPedigreeFunction'].quantile(0.75)
iqr = q3 - q1

print(q3 + 1.5 * iqr)
dpfout = df[(df['DiabetesPedigreeFunction'] > q3 + 1.5 * iqr) | (df['DiabetesPedigreeFunction'] < q1 - 1.5 * iqr)]
sns.pairplot(dpfout, hue='Outcome',palette='husl', diag_kind='hist')
sns.heatmap(dpfout.corr(), annot=True, fmt='.2f')
sns.boxplot(x=df['Age'])
q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)
iqr = q3 - q1

print(q3 + 1.5 * iqr)
ageout = df[(df['Age'] > q3 + 1.5 * iqr) | (df['Age'] < q1 - 1.5 * iqr)]
sns.pairplot(ageout, hue='Outcome',palette='husl', diag_kind='hist')
sns.heatmap(ageout.corr(), annot=True, fmt='.2f')
outlier_count = {'Pregnancies':prgout.count().max(), 'Glucose':gluout.count().max(), 'Blood Pressure':blpout.count().max(), 'Skin Thickness':sknout.count().max(), 'Insulin':insout.count().max(), 'BMI':bmiout.count().max(), 'Diabetes Pedigree Function':dpfout.count().max(), 'Age':ageout.count().max()}
outlier_count = {k: v for k, v in reversed(sorted(outlier_count.items(), key=lambda item: item[1]))}
outlier_count
outlier_percent = {}
for key in outlier_count:
    outlier_percent[key] = round(outlier_count[key] / df.count().max(),4)

outlier_percent = {k: v for k, v in reversed(sorted(outlier_percent.items(), key=lambda item: item[1]))}
outlier_percent
#Procurando por valores nulos
df.isnull().sum()
def calc_vif(tabela):
  vif = pd.DataFrame()
  vif['variaveis'] = tabela.columns
  vif['vif'] = [variance_inflation_factor(tabela.values, i) for i in range(tabela.shape[1])]

  return vif

vif = calc_vif(df)
vif.sort_values('vif', ascending=False)
forest = RandomForestClassifier(n_estimators=100,
                              random_state=0)

forest.fit(X, y)

features = forest.feature_importances_
features_and_names = {}
for i in range(len(features)):
    features_and_names[df.columns[i]] = features[i]

features_and_names = {k: v for k, v in reversed(sorted(features_and_names.items(), key=lambda item: item[1]))}
features_and_names
sns.distplot(df[df['Outcome'] == 0]['Glucose'], kde=False)
sns.distplot(df[df['Outcome'] == 1]['Glucose'], kde=False)
sns.distplot(df[df['Outcome'] == 0]['BMI'], kde=False)
sns.distplot(df[df['Outcome'] == 1]['BMI'], kde=False)
fig, ax = plt.subplots(3,3,figsize=(30,15))
for variable, i in zip(df.columns, range(len(df.columns))):
  sns.distplot(df[variable], ax=ax[i//3][i%3])
plt.show()
stats_auc = {}
stats_ks = {}
stats_recall = {}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(preds[:20])
print(y_test.head(20))

fpr, tpr, _ = roc_curve(y_test, preds)
alg = 'Random Forest'
a = auc(fpr,tpr)
ks = stats.ks_2samp(preds,y_test)
rec = recall_score(y_test, preds)
stats_auc[alg] = a
stats_ks[alg] = ks.pvalue
stats_recall[alg] = rec
print("AUC:",a)
print("KS statistic:",ks.statistic, "pvalue:", ks.pvalue)
print("Recall:",rec)
probs = model.predict_proba(X_test)
sns.distplot(probs)
xgb = XGBClassifier(n_estimators=100,learning_rate=0.05, random_state=0) 

xgb.fit(X_train, y_train)
preds = xgb.predict(X_test)

print(preds[:20])
print(y_test.head(20))

fpr, tpr, _ = roc_curve(y_test, preds)
alg = 'XGB'
a = auc(fpr,tpr)
ks = stats.ks_2samp(preds,y_test)
rec = recall_score(y_test, preds)
stats_auc[alg] = a
stats_ks[alg] = ks.pvalue
stats_recall[alg] = rec
print("AUC:",a)
print("KS statistic:",ks.statistic, "pvalue:", ks.pvalue)
print("Recall:",rec)
probs = xgb.predict_proba(X_test)
sns.distplot(probs)
lr = LogisticRegression(random_state=0,max_iter=1000000)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, preds)
alg = 'Logistic Regression'
a = auc(fpr,tpr)
ks = stats.ks_2samp(preds,y_test)
rec = recall_score(y_test, preds)
stats_auc[alg] = a
stats_ks[alg] = ks.pvalue
stats_recall[alg] = rec
print("AUC:",a)
print("KS statistic:",ks.statistic, "pvalue:", ks.pvalue)
print("Recall:",rec)
sns.distplot(preds)
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, preds)
alg = 'KNN'
a = auc(fpr,tpr)
ks = stats.ks_2samp(preds,y_test)
rec = recall_score(y_test, preds)
stats_auc[alg] = a
stats_ks[alg] = ks.pvalue
stats_recall[alg] = rec
print("AUC:",a)
print("KS statistic:",ks.statistic, "pvalue:", ks.pvalue)
print("Recall:",rec)
probs = knn.predict_proba(X_test)
sns.distplot(probs)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
preds = tree.predict(X_test)

print(preds[:20])
print(y_test.head(20))

fpr, tpr, _ = roc_curve(y_test, preds)
alg = 'Decision Tree'
a = auc(fpr,tpr)
ks = stats.ks_2samp(y_test,preds)
rec = recall_score(y_test, preds)
stats_auc[alg] = a
stats_ks[alg] = ks.pvalue
stats_recall[alg] = rec
print("AUC:",a)
print("KS statistic:",ks.statistic, "pvalue:", ks.pvalue)
print("Recall:",rec)
probs = tree.predict_proba(X_test)
sns.distplot(probs)
stats_auc = {k: v for k, v in reversed(sorted(stats_auc.items(), key=lambda item: item[1]))}
stats_auc
stats_ks = {k: v for k, v in reversed(sorted(stats_ks.items(), key=lambda item: item[1]))}
stats_ks
stats_recall = {k: v for k, v in reversed(sorted(stats_recall.items(), key=lambda item: item[1]))}
stats_recall
