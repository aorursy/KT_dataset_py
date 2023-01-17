import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings('ignore')

pima = pd.read_csv("../input/diabetes.csv")
pima.head()
pima_original = pima.copy()
pima.shape
pima.describe()
pima.isnull().sum()
pima.columns
pima.dtypes
pima.hist(figsize=(10,8))
pima["Outcome"].size
pima['Outcome'].value_counts()
pima["Outcome"].value_counts(normalize=True).plot.bar(title='Outcome')
pima["Outcome"].value_counts(normalize=True)*100
pima['Pregnancies'].value_counts()
pima["Pregnancies"].value_counts(normalize=True).plot.bar(title='Pregnancies')
pima["Pregnancies"].value_counts(normalize=True)*100
sns.distplot(pima["Age"])
sns.distplot(pima["BMI"])
sns.distplot(pima["Glucose"])
sns.distplot(pima["BloodPressure"])
sns.distplot(pima["SkinThickness"])
sns.distplot(pima["Insulin"])
sns.distplot(pima["DiabetesPedigreeFunction"])

age = pd.crosstab(pima["Age"],pima["Outcome"])
age.div(age.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(17,15))
plt.xlabel("Age")
plt.ylabel("Outcome")
plt.show()
pregnencies = pd.crosstab(pima["Pregnancies"],pima["Outcome"])
pregnencies.div(pregnencies.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(8,8))
plt.xlabel("Pregnancies")
plt.ylabel("Outcome")
plt.show()
glucose = pd.crosstab(pima["Glucose"],pima["Outcome"])
glucose.div(glucose.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(27,30))
plt.xlabel("Glucose")
plt.ylabel("Outcome")
plt.show()
bloodpressure = pd.crosstab(pima["BloodPressure"],pima["Outcome"])
bloodpressure.div(bloodpressure.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(14,7))
plt.xlabel("BloodPressure")
plt.ylabel("Outcome")
plt.show()
pima.columns
X = pima.drop('Outcome',axis=1)
X.columns
X.head(3)
y=pima['Outcome']
y.head(3)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(x_train,y_train)
print(logistic_model.score(x_train,y_train)*100)
pred_logistic=logistic_model.predict(x_test)
score_logistic = accuracy_score(pred_logistic,y_test)*100
score_logistic
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()
tree_model.fit(x_train,y_train)
print(tree_model.score(x_train,y_train))
tree_pred = tree_model.predict(x_test)
tree_score = accuracy_score(tree_pred,y_test)
tree_score
