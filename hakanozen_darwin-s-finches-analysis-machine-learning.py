import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mlxtend.plotting import category_scatter
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tools

from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))
df1 = pd.read_csv(open('../input/Fig.10-03-data-12.csv','rU'), encoding='utf-8', engine='c')
df2 = pd.read_csv(open('../input/Fig.10-03-data-75.csv','rU'), encoding='utf-8', engine='c')
df1.head()
df1.drop(["band"], axis = 1, inplace = True)
df1.dtypes
df1 = df1.convert_objects(convert_numeric=True)
df1.dtypes
df2.head()
df2.rename(index=str, columns={"Beak length, mm": "blength", "Beak depth, mm": "bdepth"}, inplace = True)
df2.drop(["band"], axis = 1, inplace = True)
df2.head()
df2.dtypes
df2 = df2.convert_objects(convert_numeric=True)
df2.dtypes
df = pd.concat([df1, df2], ignore_index=True)
df.tail()
df.drop_duplicates(subset=None, keep='first', inplace=True)
df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)
df.tail()
df = df.dropna(axis=0)
df.info()
counts = df.species.value_counts()
fig = {
    'data': [
            {'labels': ['fortis','scandens'],
            'values': [counts["fortis"], counts["scandens"]],
            'type': 'pie',
            'hoverinfo':'none',
            'textinfo':'value+percent',
            'textfont': {"size": 16}}
            ],
    'layout': {'title': 'Distribution of Species in the Dataset',
               'titlefont': {"size": 20},
               'showlegend': False,
               "annotations": [
                               {"font": {"size": 18},
                                "showarrow": False,
                                "text": "scandens",
                                "x": 0.18,
                                "y": 0.8},
                               {"font": {"size": 18},
                               "showarrow": False,
                               "text": "fortis",
                               "x": 0.79,
                               "y": 0.3}
                             ]
              }
}

py.iplot(fig)
df_fortis = df.loc[df['species'] == 'fortis']
df_scandens = df.loc[df['species'] == 'scandens']
trace0 = go.Scatter(
    x = df_fortis.bdepth,
    y = df_fortis.blength,
    name = 'fortis',
    mode = 'markers',
    marker = dict(size = 15,color = 'rgba(255, 170, 0)',line = dict(width = 2))
)

trace1 = go.Scatter(
    x = df_scandens.bdepth,
    y = df_scandens.blength,
    name = 'scandens',
    mode = 'markers',
    marker = dict(size = 15,color = 'rgba(200, 200, 255)',line = dict(width = 2))
)

data = [trace0, trace1]

layout = dict(title = 'Fortis and Scandens Species',
              xaxis = dict(title = 'Beak Depth (mm)'),
              yaxis = dict(title = 'Beak Length (mm)'),
              font=dict(size=18)
             )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Fortis and Scandens Species')
trace1 = go.Histogram(
    x=df_fortis.blength,
    histnorm='percent',
    name='beak length',
    xbins=dict(start=7,end=13,size=0.2))

trace2 = go.Histogram(
    x=df_fortis.bdepth,
    histnorm='percent',
    name='beak depth',
    xbins=dict(start=7,end=13,size=0.2))

data = [trace1, trace2]

layout = go.Layout(
    title='Sampled Histogram of Fortis Species',
    xaxis=dict(title='Long (mm)'),
    yaxis=dict(title='Percentage in the Population (%)'),
    font=dict(size=18),
    bargap=0.002
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='styled histogram')
trace1 = go.Histogram(
    x=df_scandens.blength,
    histnorm='percent',
    name='beak length',
    xbins=dict(start=7,end=16,size=0.2))

trace2 = go.Histogram(
    x=df_scandens.bdepth,
    histnorm='percent',
    name='beak depth',
    xbins=dict(start=7,end=16,size=0.2))

data = [trace1, trace2]

layout = go.Layout(
    title='Sampled Histogram of Scandens Species',
    xaxis=dict(title='Long (mm)'),
    yaxis=dict(title='Percentage in the Population (%)'),
    font=dict(size=18),
    bargap=0.1
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='styled histogram')
df.species = [1 if each == "fortis" else 0 for each in df.species]
y_data = df["species"].values
x_data = df.drop(["species"], axis = 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state=46)

y_train = y_train.flatten()
y_test = y_test.flatten()

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)
list_names = []
list_accuracy = []
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
lr.fit(x_train, y_train)
LR_accuracy = lr.score(x_test, y_test)*100
LR_accuracy = round(LR_accuracy, 2)

print("LR_accuracy is %", LR_accuracy)

list_names.append("Logistic Regression")
list_accuracy.append(LR_accuracy)
from sklearn.neighbors import KNeighborsClassifier

Knn = KNeighborsClassifier(n_neighbors = 4)
Knn.fit(x_train, y_train)
Knn_accuracy = Knn.score(x_test, y_test)*100
Knn_accuracy = round(Knn_accuracy, 2)

print("Knn_accuracy is %", Knn_accuracy)

list_names.append("KNN")
list_accuracy.append(Knn_accuracy)
from sklearn.svm import SVC

svm = SVC(random_state=1)
svm.fit(x_train, y_train)
SVM_accuracy = svm.score(x_test, y_test)*100
SVM_accuracy = round(SVM_accuracy, 2)

print("SVM_accuracy is %", SVM_accuracy)

list_names.append("SVM")
list_accuracy.append(SVM_accuracy)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)
NaiveBayes_acuracy = nb.score(x_test, y_test)*100
NaiveBayes_acuracy = round(NaiveBayes_acuracy,2)

print("NaiveBayes_acuracy is %", NaiveBayes_acuracy)

list_names.append("Naive Bayes")
list_accuracy.append(NaiveBayes_acuracy)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
DecisionTree_accuracy = dt.score(x_test, y_test)*100
DecisionTree_accuracy = round(DecisionTree_accuracy,2)

print("DecisionTree_accuracy is %", DecisionTree_accuracy)

list_names.append("Decision Tree")
list_accuracy.append(DecisionTree_accuracy)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 200, random_state = 1)
rf.fit(x_train, y_train)
RandomForest_accuracy = rf.score(x_test, y_test)*100
RandomForest_accuracy = round(RandomForest_accuracy, 2)

print("RandomForest_accuracy is %", RandomForest_accuracy)

list_names.append("Random Forest")
list_accuracy.append(RandomForest_accuracy)
df = pd.DataFrame({'METHOD': list_names, 'ACCURACY (%)': list_accuracy})
df = df.sort_values(by=['ACCURACY (%)'])
df = df.reset_index(drop=True)
df.head(6)
trace1 = go.Bar(x = df.iloc[:,0].tolist(), y = df.iloc[:,1].tolist())

data1 = [trace1]
layout1 = go.Layout(
    margin=dict(b=150),
    title='Comparison of the Learning Methods',
    xaxis=dict(titlefont=dict(size=16), tickangle=-60),
    yaxis=dict(title='ACCURACY (%)',gridwidth=1, gridcolor='#bdbdbd', range=[95, 98]),
    font=dict(size=16),
    bargap = 0.7,
    barmode='group')

fig = go.Figure(data=data1, layout=layout1)
py.iplot(fig, filename='grouped-bar')