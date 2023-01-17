# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd     #load data and data manipulation

import seaborn as slt  #high level inteface visualization

import numpy as np      #for mathematical computational

from sklearn import svm  #svm (Support Vector Machine) a classification machine learning algorithm

from sklearn.model_selection import train_test_split  #splits the dataset into training and testing data

from mlxtend.plotting import plot_decision_regions    #for plotting SVM classes

from matplotlib import pyplot as plt                  #basic for visualization
df = pd.read_csv('../input/iris-dataset/iris.data.csv')

df.head()
col = df.columns

col
df.columns=['slen','swid','plen','pwid','class']

#adding our first row to the data

df.loc[150]=col
print (df.shape)

df.head()
#checking for number of nan values in any column

df.isna().sum()
#plotting pairplot of whole data

#used to find the relation among the all columns

slt.pairplot(df , hue='class')
#increasing the default figure size of matplot

plt.figure(figsize=(8,6))

#plotting scatter plot

#hue is for categorizing our data

slt.scatterplot(df['slen'],df['swid'],data=df,hue='class',s=50)
plt.figure(figsize=(8,6))

#scatterplot between petal length and petal width

slt.scatterplot(df['plen'],df['pwid'],data=df,hue='class',s=50)
slt.lmplot(x='slen',y='swid',data=df,col='class')
slt.lmplot(x='plen',y='pwid',data=df,col='class')
plt.figure(figsize=(8,5))

#heatmap of correlation matrix of our data

#it clearly indicates that sepal width is less correlated with other attributes

#sepal length, petal length and petal width are highly correalted with each other 

slt.heatmap(df.corr(),annot=True,vmax=1,cmap='Greens')
m = pd.Series(df['class']).astype('category')

df['class']=m.cat.codes
Y=df['class'] 

X = df.drop(columns=['class'])
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3)

ytrain.index=np.arange(105)
clf = svm.SVC(gamma='auto')
pre = clf.fit(xtrain,ytrain)
clf.score(xtest,ytest)
plt.figure(figsize=(8,6))

#plotting scatter plot for each different classes

for i in range (0,pca.shape[0]):

    if ytrain[i]==0:

       c1=plt.scatter(pca[i,0],pca[i,1],c='r',marker='+',s=60)   # Iris-setosa

    elif ytrain[i]==1:

       c2=plt.scatter(pca[i,0],pca[i,1],c='b',marker='o',s=50)   # Iris-versicolor 

    elif ytrain[i]==2:

       c3=plt.scatter(pca[i,0],pca[i,1],c='g',marker='*',s=60)   # Iris-virginica



#defining legends of our plot

plt.legend([c1,c2,c3],['Iris-setosa','Iris-versicolor','Iris-viginica'])

 



x_min, x_max = pca[:, 0].min() - 1,   pca[:,0].max() + 1

y_min, y_max = pca[:, 1].min() - 1,   pca[:, 1].max() + 1



#np.meshgrid is to create a rectangular grid out of an array of x values and an array of y values.

x1, y1 = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))

#np.ravel returns contiguous flattened array (1D array)

# ex. np.array((1,2),(3,4)) => [1,2,3,4]

m = clf2.predict(np.c_[x1.ravel(),  y1.ravel()])





m = m.reshape(x1.shape)

#draw contour lines 

plt.contour(x1, y1, m)

plt.title("SVM Classifiers of 3 classes")

plt.show()