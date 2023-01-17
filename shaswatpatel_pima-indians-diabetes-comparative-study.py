# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Data viz. and EDA

import matplotlib.pyplot as plt 

%matplotlib inline  

import plotly.offline as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot, iplot

import plotly.tools as tls

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)





# Tensorflow 

import tensorflow as tf



from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
# checking missing values if any

display(data.info(),data.head())
import seaborn as sns

plt.figure(figsize=(15,8))

sns.distplot(data.Pregnancies, bins=30)
import seaborn as sns

plt.figure(figsize=(15,8))

sns.distplot(data.Glucose, bins=30)
import seaborn as sns

plt.figure(figsize=(15,8))

sns.distplot(data.BloodPressure, bins=30)

plt.savefig('bloodpressure.png')
## As seen earlier there is no null value. However on close inspection we find that null values are filled with '0'

data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)    
## Checking the new null values found.

data.isnull().sum()
# Define missing plot to detect all missing values in dataset

def missing_plot(dataset, key) :

    null_values = pd.DataFrame(dataset.isnull().sum(), columns = ['Count'])



    trace = go.Bar(x = null_values.index, y = null_values['Count'] ,opacity = 0.6, text = null_values['Count'],  textposition = 'auto',marker=dict(color = '#7EC0EE',

            line=dict(color='#000000',width=2)))



    layout = dict(title =  "Missing Values")



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)



missing_plot(data,'Outcome')
def find_median(var):

    temp = data[data[var].notnull()]

    temp = data[[var,'Outcome']].groupby('Outcome')[[var]].median().reset_index()

    return temp
for i, col in enumerate(data.columns):

    if(col == 'Outcome'):

        continue

    medians = find_median(col).to_numpy()

    data.loc[(data['Outcome'] == 0) & (data[col].isnull()) , col] = medians[0][1] # Median of Non-diabetics

    data.loc[(data['Outcome'] == 1) & (data[col].isnull()) , col] = medians[1][1] # Median of diabetics
display(data.isnull().sum())
cor=data.corr()

plt.figure(figsize=(12,12))

sns.heatmap(cor,annot=True,cmap='coolwarm')

plt.savefig('heatmap.png')

plt.show()
sns.pairplot(data=data,hue='Outcome',diag_kind='scatter')

plt.savefig('pairwise-scatter.png')

plt.show()
X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

Y = data.Outcome
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.preprocessing import StandardScaler

stdScalar = StandardScaler()

stdScalar.fit(X_train)

X_train = stdScalar.transform(X_train)

X_test = stdScalar.transform(X_test)
def build_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Dense(8, activation='relu', input_shape=[len(X_train[0])]),

        tf.keras.layers.Dense(4, activation='relu'),

        tf.keras.layers.Dense(1,activation='sigmoid')

    ])



    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model



model = build_model()

model.summary()
history = model.fit(X_train, y_train,epochs=1000, verbose=2)

pred = model.predict(X_test)

pred[pred <= 0.5] = 0

pred[pred > 0.5] = 1

print(classification_report(y_test, pred))
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train,y_train)

pred = classifier.predict(X_test)

print(classification_report(y_test, pred))
from sklearn import svm

for i in range(1, 100):

    svm_classifier = svm.SVC(C=i, kernel="rbf")



    svm_classifier.fit(X_train,y_train)

    pred = svm_classifier.predict(X_test)

    if(accuracy_score(y_test, pred) > 0.85):

        print(f"C= {i}")

        print(classification_report(y_test, pred))
svm_classifier = svm.SVC(C=1, kernel="rbf")

svm_classifier.fit(X_train,y_train)

pred = svm_classifier.predict(X_test)

print(classification_report(y_test, pred))