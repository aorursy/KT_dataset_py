!pip install -U ppscore
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import ppscore as pps



import warnings

warnings.filterwarnings("ignore")



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")

pd.options.plotting.backend = "plotly"



data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
fig = data.nunique().reset_index().plot(kind='bar', x='index', y=0, color=0)

fig.update_layout(title='Unique Value Count Plot', xaxis_title='Variables', yaxis_title='Unique value count')

fig.show()
fig = data.isnull().sum().reset_index().plot(kind='bar', x='index', y=0)

fig.update_layout(title='Missing Value Plot', xaxis_title='Variables', yaxis_title='Missing value count')

fig.show()
df = data.dtypes.value_counts().reset_index()

df['index'] = df['index'].astype('str')

sns.barplot(df['index'], df[0])

plt.title('DataType Count')

plt.xlabel('DataTypes')

plt.ylabel('Count')

plt.show()
sns.heatmap(data.corr())

plt.title('Correelation in data')

plt.show()
matrix_df = pps.matrix(data)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)

plt.title('PPS Matrix')

plt.show()
sns.FacetGrid(data, hue="DEATH_EVENT", height=6,).map(sns.kdeplot, "age",shade=True).add_legend()

plt.title('Age Distribution Plot')

plt.show()
px.box(data, x='DEATH_EVENT', y='creatinine_phosphokinase', color='smoking', title='Creatinine Phosphokinase Distribution')
px.violin(data, x='ejection_fraction', color='DEATH_EVENT', title='Ejection Fraction Distribution')
px.box(data, x='DEATH_EVENT', y='platelets', color='diabetes', points='all', title='Platelets Distribution')
px.box(data, x='DEATH_EVENT', y='time', color='smoking', notched=True, title='Time under observation Distribution')
sns.barplot(data=data, x='high_blood_pressure', y='platelets', hue='DEATH_EVENT')

plt.title('high_blood_pressure vs platelets')

plt.show()
sns.scatterplot(data=data, x='serum_creatinine', y='serum_sodium', hue='diabetes')

plt.title('serum_creatinine vs serum_sodium')

plt.show()
sns.stripplot(data=data, x="DEATH_EVENT", y="time")

plt.title('Time vs Death Event')

plt.show()
sns.pairplot(data=data[['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time','DEATH_EVENT']], hue='DEATH_EVENT')
from sklearn.preprocessing import StandardScaler



cols = ['age','creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
scaler = StandardScaler()

data[cols] = scaler.fit_transform(data[cols])
X = data.drop('DEATH_EVENT', axis=1)

y = data['DEATH_EVENT'].copy()
from sklearn.model_selection import train_test_split



train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=13)
from sklearn.metrics import roc_curve,accuracy_score,plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier(min_samples_split=2, class_weight={0:2,1:7}, random_state=13)

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Decision Tree Classifier is', accuracy_score(prediction,test_y))
plot_confusion_matrix(model, test_X, test_y)

plt.title('Decision Tree Confusion Matrix')

plt.show()
model = DecisionTreeClassifier(min_samples_split=2, class_weight={0:2,1:7}, random_state=13)

model.fit(train_X, train_y)

y_pred_prob = model.predict_proba(test_X)[:,1]

fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='DT')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Decision Tree ROC Curve')

plt.show()
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(min_samples_split=2, class_weight={0:2,1:7}, random_state=13)

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Random Forest Classifier is', accuracy_score(prediction,test_y))
plot_confusion_matrix(model, test_X, test_y)

plt.title('Random Forest Confusion Matrix')

plt.show()
model = RandomForestClassifier(min_samples_split=2, class_weight={0:2,1:7}, random_state=13)

model.fit(train_X, train_y)

y_pred_prob = model.predict_proba(test_X)[:,1]

fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='RF')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Random Forest ROC Curve')

plt.show()