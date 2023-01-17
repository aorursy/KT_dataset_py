import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"



data = pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv')

train_data = pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv')

test_data = pd.read_csv('../input/kepler-labelled-time-series-data/exoTest.csv')
data.describe()
# sns.heatmap(data.corr())

# plt.title('Correlation in the data')

# plt.show()
fig = data.isnull().sum().reset_index().plot(kind='bar', x='index', y=0)

fig.update_layout(title="Checking for Missing Values", xaxis_title="Variable", yaxis_title="Missing Value Count")

fig.show()
# checking for duplicate values



data = data.drop_duplicates()
sns.boxplot(data=data, x='LABEL', y='FLUX.1')

plt.title('Distribution of FLUX.1')

plt.show()
print('Dropping Outliers')

data.drop(data[data['FLUX.1']>250000].index, axis=0, inplace=True)
sns.FacetGrid(data, hue="LABEL", height=6,).map(sns.kdeplot, "FLUX.1",shade=True).add_legend()

plt.title('KDE Plot for FLUX.1')

plt.show()
data['LABEL'].value_counts().reset_index().plot(kind='bar', x='index', y='LABEL', color='LABEL')
sns.scatterplot(data=data, x='FLUX.1', y='FLUX.4', hue='LABEL', palette=['g','r'])

plt.title('Relation of FLUX1 and FLUX4')

plt.show()
fig = px.scatter_matrix(data[['FLUX.1','FLUX.2','FLUX.3','FLUX.4','FLUX.5']])

fig.update_layout(title="Scatter Matrix for first 5 light intensities")

fig.show()
print('Pairplot for first 5 intensities')

sns.pairplot(data=data[['LABEL','FLUX.1','FLUX.2','FLUX.3','FLUX.4','FLUX.5']], hue='LABEL')

plt.show()
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.3)



train_X = train.drop('LABEL', axis=1)

train_y = train['LABEL']

test_X = test.drop('LABEL', axis=1)

test_y = test['LABEL']
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import roc_curve,accuracy_score,plot_confusion_matrix
model = LogisticRegression(class_weight={1:100, 2:1})

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is', accuracy_score(prediction,test_y))
plot_confusion_matrix(model, test_X, test_y)

plt.title('LR Confusion Matrix')

plt.show()
model = SVC(C=0.1, kernel='poly')

model.fit(train_X,train_y)

prediction = model.predict(test_X)

print('The accuracy of the SVC is', accuracy_score(prediction,test_y))
plot_confusion_matrix(model, test_X, test_y)

plt.title('SVC Confusion Matrix')

plt.show()
model = DecisionTreeClassifier(max_depth=5, random_state=13)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

print('The accuracy of the Decision Tree is', accuracy_score(prediction,test_y))
plot_confusion_matrix(model, test_X, test_y)

plt.title('Decision Tree Confusion Matrix')

plt.show()
model = DecisionTreeClassifier(max_depth=5, random_state=13)

model.fit(train_X, train_y)

y_pred_prob = model.predict_proba(test_X)[:,1]

fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob, pos_label=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='DT')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Decision Tree ROC Curve')

plt.show()
from imblearn.over_sampling import SMOTE

model = SMOTE()

ov_train_x,ov_train_y = model.fit_sample(data.drop('LABEL',axis=1), data['LABEL'])

ov_train_y = ov_train_y.astype('int')
ov_train_y.value_counts().reset_index().plot(kind='bar', x='index', y='LABEL')
train_X, test_X, train_y, test_y = train_test_split(ov_train_x, ov_train_y, test_size=0.33, random_state=42)



model = DecisionTreeClassifier(max_depth=5, random_state=13)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

print('The accuracy of the Decision Tree is', accuracy_score(prediction,test_y))
plot_confusion_matrix(model, test_X, test_y)

plt.title('Decision Tree Confusion Matrix')

plt.show()
model = DecisionTreeClassifier(max_depth=5, random_state=13)

model.fit(train_X, train_y)

y_pred_prob = model.predict_proba(test_X)[:,1]

fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob, pos_label=2)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='DT')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Decision Tree ROC Curve')

plt.show()