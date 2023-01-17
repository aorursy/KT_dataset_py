import numpy as np
import random
import pandas as pd
from pandas.tools import plotting
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)  
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
# from lightgbm import LGBMClassifier
from sklearn.metrics import  accuracy_score

import xgboost as xgb
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler, LabelBinarizer
# auxiliary function
from sklearn.preprocessing import LabelEncoder
def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/Iris.csv')
table = ff.create_table(df.head())
py.iplot(table,filename='jupyter-table1')
py.iplot(ff.create_table(df.describe()),filename='jupyter-table1')
df.info()
Species = df['Species'].unique()
Species
species_count = df['Species'].value_counts()
data = [go.Bar(
    x = species_count.index,
    y = species_count.values,
    marker = dict(color = random_colors(3))
)]
py.iplot(data)
corelation = df.corr()
data = [go.Heatmap(z = np.array(corelation.values),
                   x = np.array(corelation.columns),
                   y = np.array(corelation.columns),
                     colorscale='Blackbody',)
       ]
py.iplot(data)
setosa = go.Scatter(x = df['SepalLengthCm'][df.Species =='Iris-setosa'], y = df['SepalWidthCm'][df.Species =='Iris-setosa']
                   , mode = 'markers', name = 'setosa')
versicolor = go.Scatter(x = df['SepalLengthCm'][df.Species =='Iris-versicolor'], y = df['SepalWidthCm'][df.Species =='Iris-versicolor']
                   , mode = 'markers', name = 'versicolor')
virginica = go.Scatter(x = df['SepalLengthCm'][df.Species =='Iris-virginica'], y = df['SepalWidthCm'][df.Species =='Iris-virginica']
                   , mode = 'markers', name = 'virginica')
data = [setosa, versicolor, virginica]
fig = dict(data=data)
py.iplot(fig, filename='styled-scatter')
setosa = go.Scatter(x = df['PetalLengthCm'][df.Species =='Iris-setosa'], y = df['PetalWidthCm'][df.Species =='Iris-setosa']
                   , mode = 'markers', name = 'setosa')
versicolor = go.Scatter(x = df['PetalLengthCm'][df.Species =='Iris-versicolor'], y = df['PetalWidthCm'][df.Species =='Iris-versicolor']
                   , mode = 'markers', name = 'versicolor')
virginica = go.Scatter(x = df['PetalLengthCm'][df.Species =='Iris-virginica'], y = df['PetalWidthCm'][df.Species =='Iris-virginica']
                   , mode = 'markers', name = 'virginica')
data = [setosa, versicolor, virginica]
fig = dict(data=data)
py.iplot(fig, filename='styled-scatter')
trace0 = go.Box(y=df['PetalWidthCm'][df['Species'] == 'Iris-setosa'],boxmean=True, name = 'setosa')

trace1 = go.Box(y=df['PetalWidthCm'][df['Species'] == 'Iris-versicolor'],boxmean=True, name = 'versicolor')

trace2 = go.Box(y=df['PetalWidthCm'][df['Species'] == 'Iris-virginica'],boxmean=True, name = 'virginica')

data = [trace0, trace1, trace2]
py.iplot(data)
trace0 = go.Box(y=df['PetalLengthCm'][df['Species'] == 'Iris-setosa'],name = 'setosa')

trace1 = go.Box(y=df['PetalLengthCm'][df['Species'] == 'Iris-versicolor'], name = 'versicolor')

trace2 = go.Box(y=df['PetalLengthCm'][df['Species'] == 'Iris-virginica'], name = 'virginica')

data = [trace0, trace1, trace2]
py.iplot(data)
trace0 = go.Box(y=df['SepalLengthCm'][df['Species'] == 'Iris-setosa'], name = 'setosa')

trace1 = go.Box(y=df['SepalLengthCm'][df['Species'] == 'Iris-versicolor'], name = 'versicolor')

trace2 = go.Box(y=df['SepalLengthCm'][df['Species'] == 'Iris-virginica'], name = 'virginica')

data = [trace0, trace1, trace2]
py.iplot(data)
setosa = go.Box(y=df['SepalWidthCm'][df['Species'] == 'Iris-setosa'])

versicolor = go.Box(y=df['SepalWidthCm'][df['Species'] == 'Iris-versicolor'])

virginica = go.Box(y=df['SepalWidthCm'][df['Species'] == 'Iris-virginica'])

data = [trace0, trace1, trace2]
py.iplot(data)
plt.subplots(figsize = (10,8))

plotting.andrews_curves(df.drop("Id", axis=1), "Species")
g = sns.lmplot(x="SepalWidthCm", y="SepalLengthCm", hue="Species", data=df)
g = sns.lmplot(x="PetalWidthCm", y="PetalLengthCm", hue="Species", data=df)
x = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)
lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
lr_predict = lr_model.predict(x_test)
print('Logistic Regression - ',accuracy_score(lr_predict,y_test))
svm_model = SVC(kernel='linear')
svm_model.fit(x_train,y_train)
svc_predict = svm_model.predict(x_test)
print('SVM - ',accuracy_score(lr_predict,y_test))
nb_model = GaussianNB()
nb_model.fit(x_train,y_train)
nb_predict = nb_model.predict(x_test)
print('Naive bayes - ',accuracy_score(nb_predict,y_test))
dt_model = DecisionTreeClassifier(max_leaf_nodes=3)
dt_model.fit(x_train,y_train)
dt_predict = dt_model.predict(x_test)
print('Decision Tree - ',accuracy_score(dt_predict,y_test))
rfc_model = RandomForestClassifier(max_depth=3)
rfc_model.fit(x_train,y_train)
rfc_predict = rfc_model.predict(x_test)
print('Random Forest - ',accuracy_score(rfc_predict,y_test))
etc_model = ExtraTreesClassifier()
etc_model.fit(x_train,y_train)
etc_predict = etc_model.predict(x_test)
print('Extra Tree Classifier - ',accuracy_score(etc_predict,y_test))

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train,y_train)
knn_predict = knn_model.predict(x_test)
print('knn - ',accuracy_score(knn_predict,y_test))

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelBinarizer
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

X = StandardScaler().fit_transform(X)
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
shallow_model = Sequential()
shallow_model.add(Dense( 4, input_dim=4, activation = 'relu'))
shallow_model.add(Dense( units = 10, activation= 'relu'))
shallow_model.add(Dense( units = 3, activation= 'softmax'))
shallow_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
shallow_history = shallow_model.fit(x_train, y_train, epochs = 150, validation_data = (x_test, y_test))
plt.plot(shallow_history.history['acc'])
plt.plot(shallow_history.history['val_acc'])
plt.title("Accuracy")
plt.legend(['train', 'test'])
plt.show()
plt.plot(shallow_history.history['loss'])
plt.plot(shallow_history.history['val_loss'])
plt.plot('Loss')
plt.legend(['Train','Test'])
plt.show()
deep_model = Sequential()
deep_model.add(Dense( 4, input_dim=4, activation = 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 3, activation= 'softmax'))
deep_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
deep_history = deep_model.fit(x_train, y_train, epochs = 150, validation_data = (x_test, y_test))
plt.plot(deep_history.history['acc'])
plt.plot(deep_history.history['val_acc'])
plt.title("Accuracy")
plt.legend(['train', 'test'])
plt.show()
plt.plot(deep_history.history['loss'])
plt.plot(deep_history.history['val_loss'])
plt.plot('Loss')
plt.legend(['Train','Test'])
plt.show()
