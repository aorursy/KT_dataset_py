#importing necessary libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools

import plotly.express as px

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import chart_studio.plotly as py #first install the chart studio
!pip install chart_studio #installing chart studio
data = pd.read_csv("../input/mushroom-classification/mushrooms.csv") #importing dataset
data.head(5) #Taking a Glimpse of the Dataset
data.isnull().sum() #Chekcing for the Null Values
#Checking for Class Count



print(data[data['class'] == 'p'].count())

print("-----------------------------------")

print(data[data['class'] == 'e'].count())
#Setting up the parcats dimension for the parallel categories diagram



class_dim = go.parcats.Dimension(

    values = data['class'].values,

    label = "Mushroom Types",

    categoryarray = ['p', 'e'],

    ticktext = ['poisonous', 'edible']

)



cap_shape_dim = go.parcats.Dimension(

    values = data['cap-shape'].values,

    label = "Cap Shape"



)



cap_surface_dim = go.parcats.Dimension(

    values = data['cap-surface'].values,

    label = "Cap Surface"

)



cap_color_dim = go.parcats.Dimension(

    values=data['cap-color'].values,

    label = "Cap Color",

    categoryarray = ['n', 'y', 'w', 'g', 'e', 'p', 'b', 'u', 'c', 'r'],

    ticktext = ['brown', 'yellow', 'white', 'gray', 'red', 'pink', 'buff', 'purple', 'cinnamon', 'green']

)



#Creating a parcats trace using go Parcats

color = [1 if i =="e" else 0 for i in data['class']]

colorscale = [[0, 'lightsteelblue'], [1, 'mediumseagreen']];



data_ = [

    go.Parcats(

    dimensions=[class_dim, cap_shape_dim, cap_surface_dim, cap_color_dim],

    line = {'color':color,

           'colorscale': colorscale},

    hoveron='dimension', hoverinfo='count+probability',

    arrangement = 'freeform',

    labelfont = {'size':15},

    tickfont = {'size': 13})

]



iplot(data_)
le = LabelEncoder()
data = data.apply(le.fit_transform)
data.head(10)
X = data.iloc[:,1:].values
y = data.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
import keras
from keras.models import Sequential

from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 22))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 15, epochs = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt = 'd')
total = np.sum([cm[0][0], cm[1][1], cm[0][1], cm[1][0]])

accuracy = np.sum([cm[0][0], cm[1][1]])/total*100

print("There is {} of accuracy".format(round(accuracy)))