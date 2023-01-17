# for basic operations

import numpy as np

import pandas as pd



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# for interactive visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)



# for providing path

import os

print(os.listdir('../input'))
data = pd.read_csv('../input/data.csv')



# getting the shape

print(data.shape)
# checking the  head of the dataset



data.head()
# removing the last column as it is empty



data = data.drop(['id', 'Unnamed: 32'], axis = 1)



print(data.columns)
data.describe()
# checking if there is any null values in the data



data.isnull().sum().sum()
# checking the different values contained in the diagnosis column



diagnosis = data['diagnosis'].value_counts()



diagnosis_label = diagnosis.index

diagnosis_size = diagnosis.values



colors = ['pink', 'lightgreen']



trace = go.Pie(labels = diagnosis_label,

              values = diagnosis_size,

               marker = dict(colors = colors),

               name = 'Breast Cancer',

               hole = 0.3

              )

df = [trace]

layout = go.Layout(title = 'Distribution of Patients')



fig = go.Figure(data = df, layout = layout)

py.iplot(fig)
y = data['diagnosis']

x = data.drop('diagnosis', axis = 1)



x = (x - x.mean()) / (x.std()) 

df = pd.concat([y, x.iloc[:,0:10]], axis=1)

df = pd.melt(df, id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(15, 10))

sns.violinplot(x="features", y="value", hue="diagnosis", data=df,split=True, inner="quart", palette = 'cool')

plt.title('Mean Features vs Diagnosis', fontsize = 20)

plt.xticks(rotation=90)

plt.show()
y = data['diagnosis']

x = data.drop('diagnosis', axis = 1)



x = (x - x.mean()) / (x.std()) 

df = pd.concat([y, x.iloc[:,10:20]], axis=1)

df = pd.melt(df, id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(15, 10))

sns.boxplot(x="features", y="value", hue="diagnosis", data=df, palette = 'summer')

plt.title('SE Features vs Diagnosis', fontsize = 20)

plt.xticks(rotation=90)

plt.show()
y = data['diagnosis']

x = data.drop('diagnosis', axis = 1)



x = (x - x.mean()) / (x.std()) 

df = pd.concat([y, x.iloc[:,20:30]], axis=1)

df = pd.melt(df, id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(15, 10))

sns.boxenplot(x="features", y="value", hue="diagnosis", data=df, palette = 'winter')

plt.title('Worst Features vs Diagnosis', fontsize = 20)

plt.xticks(rotation=90)

plt.show()
plt.rcParams['figure.figsize'] = (18, 15)



sns.heatmap(data.corr(), cmap = 'pink', annot = True, linewidths = 0.5, fmt = '.1f')

plt.title('Heat Map for Correlations', fontsize = 20)

plt.show()
list_to_delete = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean',

                  'radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst',

                  'concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']

x = x.drop(list_to_delete, axis = 1)



plt.rcParams['figure.figsize'] = (18, 15)

sns.heatmap(x.corr(), annot = True, cmap = 'autumn')

plt.title('Heat Map for the Reduced Data', fontsize = 20)

plt.show()
# label encoding of the dependent variable



# importing label encoder

from sklearn.preprocessing import LabelEncoder



# performing label encoding

le = LabelEncoder()

y= le.fit_transform(y)
#splitting the dataset into training and testing sets



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 16)



print("Shape of x_train :", x_train.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_test :", y_test.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# creating a model

model = RandomForestClassifier(n_estimators = 400, max_depth = 10)



# feeding the training set into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# Calculating the accuracies

print("Training accuracy :", model.score(x_train, y_train))

print("Testing accuarcy :", model.score(x_test, y_test))



# classification report

cr = classification_report(y_test, y_pred)

print(cr)



# confusion matrix 

cm = confusion_matrix(y_test, y_pred)

plt.rcParams['figure.figsize'] = (5, 5)

sns.heatmap(cm, annot = True, cmap = 'winter')

plt.title('Confusion Matrix', fontsize = 20)

plt.show()
import warnings

warnings.filterwarnings('ignore')



from sklearn.feature_selection import RFECV



# The "accuracy" scoring is proportional to the number of correct classifications

model = RandomForestClassifier() 

rfecv = RFECV(estimator = model, step = 1, cv = 5, scoring = 'accuracy')

rfecv = rfecv.fit(x_train, y_train)



print('Optimal number of features :', rfecv.n_features_)

print('Best features :', x_train.columns[rfecv.support_])
y_pred = rfecv.predict(x_test)



print("Training Accuracy :", rfecv.score(x_train, y_train))

print("Testing Accuracy :", rfecv.score(x_test, y_test))



cm = confusion_matrix(y_pred, y_test)

plt.rcParams['figure.figsize'] = (5, 5)

sns.heatmap(cm, annot = True, cmap = 'copper')

plt.show()