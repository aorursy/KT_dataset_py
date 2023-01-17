import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px

from pandas_profiling import ProfileReport

from plotly.offline import iplot

!pip install joypy

import joypy

from sklearn.cluster import KMeans



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")



data = pd.read_csv('../input/iris/Iris.csv')
data.head()
data.drop('Id', axis=1).describe()
ax = sns.boxplot(data=data, x="Species", y="SepalLengthCm")

plt.title('Box Plot')

plt.show()
sns.violinplot(data=data, x="Species", y="SepalLengthCm", size=8)

plt.title('Violin Plot')

plt.show()
sns.FacetGrid(data, hue="Species", height=6,).map(sns.kdeplot, "SepalLengthCm",shade=True).add_legend()

plt.title('KDE Plot')

plt.show()
sns.stripplot(data=data, x="Species", y="SepalLengthCm")

plt.title('Strip Plot')

plt.show()
ax = sns.boxplot(data=data, x="Species", y="SepalWidthCm")

plt.title('Box Plot')

plt.show()
ax = sns.violinplot(data=data, x="Species", y="SepalWidthCm")

plt.title('Violin Plot')

plt.show()
ax = sns.stripplot(data=data, x="Species", y="SepalWidthCm")

plt.title('Strip Plot')

plt.show()
sns.FacetGrid(data, hue="Species", height=6,).map(sns.kdeplot, "SepalWidthCm",shade=True).add_legend()

plt.title('KDE Plot')

plt.show()
ax = sns.boxplot(data=data, x="Species", y="PetalLengthCm")

plt.title('Box Plot')

plt.show()
ax = sns.violinplot(data=data, x="Species", y="PetalLengthCm")

plt.title('Violin Plot')

plt.show()
ax = sns.stripplot(data=data, x="Species", y="PetalLengthCm")

plt.title('Strip Plot')

plt.show()
sns.FacetGrid(data, hue="Species", height=6,).map(sns.kdeplot, "PetalLengthCm",shade=True).add_legend()

plt.title('KDE Plot')

plt.show()
ax = sns.boxplot(data=data, x="Species", y="PetalWidthCm")

plt.title('Box Plot')

plt.show()
ax = sns.violinplot(data=data, x="Species", y="PetalWidthCm")

plt.title('Violin Plot')

plt.show()
ax = sns.stripplot(data=data, x="Species", y="PetalWidthCm")

plt.title('Strip Plot')

plt.show()
sns.FacetGrid(data, hue="Species", height=6,).map(sns.kdeplot, "PetalWidthCm",shade=True).add_legend()

plt.title('KDE Plot')

plt.show()
sns.pairplot(data=data.drop('Id',axis=1), hue="Species", height=3, diag_kind="hist")

plt.show()
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.3)



train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

train_y = train['Species']

test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

test_y = test['Species']
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import roc_curve,accuracy_score,plot_confusion_matrix
model = LogisticRegression()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is', accuracy_score(prediction,test_y))
plot_confusion_matrix(model, test_X, test_y)

plt.title('LR Confusion Matrix')

plt.show()
model = LogisticRegression()

model.fit(train_X, train_y)

y_pred_prob = model.predict_proba(test_X)[:,1]

fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob, pos_label='Iris-versicolor')

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='Logistic Regression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Logistic Regression ROC Curve')

plt.show()
model = SVC(C=0.1, kernel='poly')

model.fit(train_X,train_y)

prediction = model.predict(test_X)

print('The accuracy of the SVC is', accuracy_score(prediction,test_y))
plot_confusion_matrix(model, test_X, test_y)

plt.title('SVC Confusion Matrix')

plt.show()
model = SVC(C=0.1, kernel='poly', probability=True)

model.fit(train_X, train_y)

y_pred_prob = model.predict_proba(test_X)[:,1]

fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob, pos_label='Iris-versicolor')

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='SVC')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('SVC ROC Curve')

plt.show()
model = KNeighborsClassifier(n_neighbors=2)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

print('The accuracy of the KNeighbors Classifier is', accuracy_score(prediction,test_y))
plot_confusion_matrix(model, test_X, test_y)

plt.title('KNN Confusion Matrix')

plt.show()
model = KNeighborsClassifier(n_neighbors=2)

model.fit(train_X, train_y)

y_pred_prob = model.predict_proba(test_X)[:,1]

fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob, pos_label='Iris-versicolor')

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='KNN')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('KNN ROC Curve')

plt.show()
model = DecisionTreeClassifier(max_depth=5, random_state=13)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

print('The accuracy of the SVC is', accuracy_score(prediction,test_y))
plot_confusion_matrix(model, test_X, test_y)

plt.title('Decision Tree Confusion Matrix')

plt.show()
model = DecisionTreeClassifier(max_depth=5, random_state=13)

model.fit(train_X, train_y)

y_pred_prob = model.predict_proba(test_X)[:,1]

fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob, pos_label='Iris-virginica')

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='DT')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Decision Tree ROC Curve')

plt.show()