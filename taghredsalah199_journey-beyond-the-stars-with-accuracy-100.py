import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_stars= pd.read_csv('../input/star-dataset/6 class csv.csv')

df_stars
df_stars['Star color'].value_counts()
df_stars['Spectral Class'].value_counts()
figure= plt.figure(figsize=(10,10))

sns.heatmap(df_stars.corr(), annot=True)



figure= plt.figure(figsize=(10,10))

sns.distplot(df_stars['Temperature (K)'])
figure= plt.figure(figsize=(20,10))

sns.boxenplot(x='Star color',y='Temperature (K)',data=df_stars)
figure= plt.figure(figsize=(20,10))

sns.boxenplot(x='Spectral Class',y='Luminosity(L/Lo)',data=df_stars,palette='winter')
sns.jointplot(x='Absolute magnitude(Mv)',y='Temperature (K)',data=df_stars, kind='hex')
sns.lmplot(x='Luminosity(L/Lo)',y='Temperature (K)',data=df_stars)
Spectral = pd.get_dummies(df_stars['Spectral Class'],drop_first=True)

df_stars.drop('Spectral Class',axis=1,inplace=True)
df_stars = pd.concat([df_stars,Spectral],axis=1)

df_stars=df_stars.drop('Star color',axis=1)
df_stars.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_stars.drop('Star type',axis=1),df_stars['Star type'],test_size=0.30,random_state=101)
from sklearn.tree import DecisionTreeClassifier

dtree= DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions= dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))

print('__________________________________________________________________________')

print(classification_report(y_test,predictions))
!pip install --upgrade scikit-learn==0.20.3
# Libraries importing

from IPython.display import Image

from sklearn.externals.six import StringIO

from sklearn.tree import export_graphviz

import pydot
# preparing for export_graphviz method

df_feat=df_stars.drop('Star type',axis=1)

features= list(df_feat.columns)

dot_data=StringIO()

export_graphviz(dtree,out_file=dot_data,feature_names=features,filled=True,rounded=True,)
#Visualize the tree in image

graph = pydot.graph_from_dot_data(dot_data.getvalue())  

Image(graph[0].create_png())
