import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from scipy.stats import norm

from scipy import stats

from sklearn.metrics import accuracy_score, confusion_matrix



%matplotlib inline

warnings.simplefilter("ignore")
data = pd.read_csv("../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv")

data.head()
data.describe()
print("Rows and Columns in the given dataset are", data.shape[0], "&", data.shape[1], "respectively")
data.isnull().any()
data['diagnosis'].value_counts()
sns.countplot(x="diagnosis", data=data)
sns.pairplot(data, hue="diagnosis", palette='twilight_r')
sns.boxenplot(data)
#individual distribuition of length and width

data.hist(linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(14,6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(3,3,1)

sns.violinplot(x='diagnosis',y='mean_area',data=data)

plt.subplot(3,3,2)

sns.violinplot(x='diagnosis',y='mean_perimeter',data=data)

plt.subplot(3,3,3)

sns.violinplot(x='diagnosis',y='mean_radius',data=data)

plt.subplot(3,3,4)

sns.violinplot(x='diagnosis',y='mean_smoothness',data=data)

plt.subplot(3,3,5)

sns.violinplot(x='diagnosis',y='mean_texture',data=data)
plt.figure(figsize=(7,4)) 

sns.heatmap(data.corr(),annot=True,cmap='cubehelix_r')

plt.show()
#checking the distribution via histograms and probability distribution

plt.figure(figsize=(15,10))

plt.subplot(5,2,1)

sns.distplot(data['mean_area'], fit=norm)

plt.subplot(5,2,2)

stats.probplot(data['mean_area'], dist = 'norm', plot = plt)

plt.subplot(5,2,3)

sns.distplot(data['mean_perimeter'], fit=norm)

plt.subplot(5,2,4)

stats.probplot(data['mean_perimeter'], dist = 'norm', plot = plt)

plt.subplot(5,2,5)

sns.distplot(data['mean_radius'], fit=norm)

plt.subplot(5,2,6)

stats.probplot(data['mean_radius'], dist = 'norm', plot = plt)

plt.subplot(5,2,7)

sns.distplot(data['mean_smoothness'], fit=norm)

plt.subplot(5,2,8)

stats.probplot(data['mean_smoothness'], dist = 'norm', plot = plt)

plt.subplot(5,2,9)

sns.distplot(data['mean_texture'], fit=norm)

plt.subplot(5,2,10)

stats.probplot(data['mean_texture'], dist = 'norm', plot = plt)

plt.show()
X = data.drop('diagnosis', axis = 1)

y = data['diagnosis']
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
#logistic regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_val)

accu_lr = accuracy_score(y_val,y_pred)

print("Accuracy score using Logistics Regression:", accu_lr*100)
#random forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_val)

accu_reg = accuracy_score(y_val,y_pred)

print("Accuracy score using Random Forest:", accu_reg*100)
#random forest

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(splitter='best', criterion="entropy")

dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_val)

accu_dtc = accuracy_score(y_val,y_pred)

print("Accuracy score using Random Forest:", accu_dtc*100)
models = pd.DataFrame({

    'Model': ['Logistic Regression','Random Forest','Decision Tree'],

    'Score': [accu_lr*100, accu_reg*100, accu_dtc*100]})

models.sort_values(by='Score', ascending=False)