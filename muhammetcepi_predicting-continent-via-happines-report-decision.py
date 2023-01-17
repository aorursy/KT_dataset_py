# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/world-happiness/2015.csv")

df.head()
df.drop(columns=["Country","Happiness Rank","Happiness Score"],inplace=True)

df.head()
df["Region"].value_counts()
my_list=list()

for index in df["Region"]:

    if "Africa" in index:

        my_list.append("Africa")

    elif "Europe" in index:

        my_list.append("Europe")

    elif "Antartica"in index:

        my_list.append("Antartica")

    elif "Australia"in index:

        my_list.append("Australia")

    elif "North America"in index:

        my_list.append("North America")

    elif "Latin America" in index:

        my_list.append("South America")

    elif "Asia" in index:

        my_list.append("Asia")

df["Region"]=my_list

df.head()
X = df[["Standard Error","Economy (GDP per Capita)","Family","Health (Life Expectancy)","Freedom","Trust (Government Corruption)","Generosity","Dystopia Residual"]].values

X[0:5]
y = df["Region"].values

y[0:5]
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

ContinentTree = DecisionTreeClassifier(criterion="entropy", max_depth = 5)

ContinentTree # it shows the default parameters
ContinentTree.fit(X_trainset,y_trainset)
ContinentTree.get_depth()
predTree = ContinentTree.predict(X_testset)
print (predTree [0:5])

print (y_testset [0:5])
from sklearn import metrics

import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
!pip install pydotplus
from sklearn.externals.six import StringIO

import matplotlib.image as mpimg

from sklearn import tree

%matplotlib inline

import pydotplus
dot_data = StringIO()

filename = "drugtree.png"

featureNames = df.columns[1:]

targetNames = df["Region"].unique().tolist()

out=tree.export_graphviz(ContinentTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png(filename)

img = mpimg.imread(filename)

plt.figure(figsize=(100, 200))

plt.imshow(img,interpolation='nearest')