#We start with importing the required packages for data processing and visualising

import numpy as np   ## linear algebra

import random         

import pandas as pd    # data processing, CSV file I/O 

from pandas.tools import plotting   #visualization

import seaborn as sns                #visualization

import matplotlib.pyplot as plt    #visualization

%matplotlib inline



import plotly.offline as py         #visualization

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected=True)  

import plotly.figure_factory as ff

#For importing the classification models

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMClassifier

from sklearn.metrics import  accuracy_score



import xgboost as xgb

import lightgbm as  lgb

from xgboost.sklearn import XGBClassifier

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
#Loading the dataset

df = pd.read_csv('../input/Iris.csv')

df.head()

table = ff.create_table(df.head())

py.iplot(table,filename='jupyter-table1')
py.iplot(ff.create_table(df.describe()),filename='jupyter-table1')
df.info()
#Types of  Species

Species = df['Species'].unique()

Species
#Andrew Curves

from matplotlib import cm

plt.subplots(figsize = (10,8))

cmap = cm.get_cmap('rainbow')

plotting.andrews_curves(df.drop("Id", axis=1), "Species",colormap=cmap)
#STRIP PLOT

fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.stripplot(x='Species',y='SepalLengthCm',data=df,jitter=True,edgecolor='gray',size=8,palette='summer',orient='v')
#Statistical Summary of the data

df.describe().plot(kind = "area",fontsize=27, figsize = (20,8), table = True,colormap="rainbow")

plt.xlabel('Statistics',)

plt.ylabel('Value')

plt.title("General Statistics of Iris Dataset")
# Histograms

df.hist(edgecolor='red', linewidth=1.4)
#pie plot

df['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8))

plt.show()
#SCATTERPPLOT(or pair plot)

sns.pairplot(df,hue='Species')
x = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y = df['Species']
encoder = LabelEncoder()

y = encoder.fit_transform(y)

y
#Splitting the dataset into training and testing dataset

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)
#LOGISTIC REGRESSION

model = LogisticRegression()

model.fit(x_train,y_train)

predict =model.predict(x_test)

print('Accuracy score of Logistic Regression - ',accuracy_score(predict,y_test))
#NAIVE BAYES

model = GaussianNB()

model.fit(x_train,y_train)

predict =model.predict(x_test)

print('Accuracy score of Naive bayes - ',accuracy_score(predict,y_test))
#SUPPORT VECTOR MACHINE

svm_model = SVC(kernel='linear')

svm_model.fit(x_train,y_train)

svc_predict = svm_model.predict(x_test)

print('Accuracy score of SVM - ',accuracy_score(svc_predict,y_test))
#DECISION TREE

dt_model = DecisionTreeClassifier(max_leaf_nodes=3)

dt_model.fit(x_train,y_train)

dt_predict = dt_model.predict(x_test)

print('Accuracy score of Decision Tree - ',accuracy_score(dt_predict,y_test))
#RANDOM FOREST

rfc_model = RandomForestClassifier(max_depth=3)

rfc_model.fit(x_train,y_train)

rfc_predict = rfc_model.predict(x_test)

print('Accuracy score of Random Forest - ',accuracy_score(rfc_predict,y_test))
#KNN 

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(x_train,y_train)

knn_predict = knn_model.predict(x_test)

print('Accuracy score of knn - ',accuracy_score(knn_predict,y_test))
#XGBoost

xg_model = xgb.XGBClassifier()

xg_model = xg_model.fit(x_train,y_train)

xg_model.score(x_test, y_test)
#TRAINING THE MODEL

#Importing packages for Deep Learning

import keras

from keras.models import Sequential

from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelBinarizer

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

y = df['Species']



X = StandardScaler().fit_transform(X)

y = LabelBinarizer().fit_transform(y)
#Splitting the dataset into training and testing set in the ration of 7:3

from sklearn.preprocessing import StandardScaler, LabelBinarizer

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

y = df['Species']



X = StandardScaler().fit_transform(X)

y = LabelBinarizer().fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
#SHALLOW Deep Learning

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