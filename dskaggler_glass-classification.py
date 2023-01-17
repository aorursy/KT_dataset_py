

#Glass classification data set kaggle
from sklearn import datasets,tree,preprocessing,grid_search

import graphviz

import pandas as pd

from scipy.stats import skew

import matplotlib.pylab as plt

from IPython.display import Image  
df=pd.read_csv("../input/glass.csv")

df.head(10)
df.isnull().sum()
na= preprocessing.scale(df['Na']) 

skness = skew(na)
natest=df['Na']

skness1 = skew(natest)

figure = plt.figure()

figure.add_subplot(121)   

plt.hist(natest,facecolor='blue',alpha=0.75) 

plt.xlabel("natest - Transformed") 

plt.title("Transformed natest Histogram") 

plt.text(25,22,"Skewness: {0:.2f}".format(skness1))

plt.show()
figure = plt.figure()

figure.add_subplot(121)   

plt.hist(na,facecolor='blue',alpha=0.75) 

plt.xlabel("na - Transformed") 

plt.title("Transformed na Histogram") 

plt.text(6,20,"Skewness: {0:.2f}".format(skness))

plt.show()
df.describe()
dfX = df.drop(['Type','RI','Fe'], axis=1)

dfX.head(5)
#normalized data

nd= preprocessing.scale(dfX) 

nddf=pd.DataFrame(nd)

nddf.head(10)
dfY=df['Type']

#dfY=dfY[:,None]

dfY.head(5)
clf=tree.DecisionTreeClassifier()

clf=clf.fit(dfX,dfY)
clf.predict([[1.64,4.49,1.10,3.63,0.06,0.0,0.00]])
dot_data=tree.export_graphviz(clf,out_file='tree.dot')     