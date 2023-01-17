# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Iris.csv')
df.head()
df.describe()
import matplotlib.pyplot as plt
plt.figure(1,figsize=(25,5))
plt.subplot(131)
plt.bar(['Setosa','Versicolor','Virginica'],[df.groupby("Species").size()[0],df.groupby("Species").size()[1],df.groupby("Species").size()[2]])
import seaborn as sns; sns.set(style="ticks",color_codes=True)
sns.FacetGrid(df,hue="Species",size=5).map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend()
sns.FacetGrid(df,hue="Species",size=5).map(plt.scatter,"PetalLengthCm","PetalWidthCm").add_legend()
test = df.drop(['Id','Species'], axis=1)
print(test)
from sklearn import tree
clf= tree.DecisionTreeClassifier().fit(test,df.Species)
import graphviz
dot_data= tree.export_graphviz(clf, out_file=None,feature_names=list(test), class_names=df.Species,filled=True,rounded=True,special_characters=True)
graph=graphviz.Source(dot_data)
graph
from sklearn.cross_validation import train_test_split
trainx,testx,trainy,testy=train_test_split(test, df.Species, test_size=0.25,random_state=3)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
clf=DecisionTreeClassifier(random_state=0)
clf.fit(trainx,trainy)

predy = clf.predict(testx)
cm=confusion_matrix(testy,predy)
cm
accuracy=accuracy_score(testy,predy)*100
print(accuracy)
from sklearn.ensemble import RandomForestClassifier
roclofo=RandomForestClassifier(random_state=0)
roclofo.fit(trainx,trainy)

ropredy = roclofo.predict(testx)
rcm=confusion_matrix(testy,ropredy)
rcm
accuracyforest=accuracy_score(testy,ropredy)*100
print(accuracyforest)
from sklearn import neighbors
konono=neighbors.KNeighborsClassifier()
konono.fit(trainx,trainy)
kononopredy=konono.predict(testx)
kononocm=confusion_matrix(testy,kononopredy)
kononocm
neighboraccuracy=accuracy_score(testy,kononopredy)*100
print(neighboraccuracy)
print("Decision tree: "+str(accuracy)+"%    <==== El mejor ｡.｡:+♡*♥") 
print("Random Forest: "+str(accuracyforest)+"%")
print("KNN: "+str(neighboraccuracy)+"%")