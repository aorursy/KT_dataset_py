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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles

from sklearn import linear_model

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
X,Y=make_circles(noise=0.1, factor=0.2, random_state=1)

X1 = X[:, 0].flatten()

X2 = X[:, 1].flatten()
#Standard Scaler

scaler = StandardScaler()

XS = scaler.fit_transform(X)



#Polynomial Features

from sklearn.preprocessing import PolynomialFeatures

PolyFeatures = PolynomialFeatures(2)



X_NEW = PolyFeatures.fit_transform(XS)


clf = linear_model.SGDClassifier(loss='log', random_state=123)

clf.fit(X_NEW, Y)

predict_Y = clf.predict(X_NEW)

#prob_Y  = clf.predict_proba(X_NEW=X_NEW) 



success = 1-sum(abs(predict_Y - Y))/len(Y)

print("Hypothesis prediction success rate is %.2f" %success)

print("Classifier Score",clf.score(X_NEW,Y))

print("Accuracy Score", accuracy_score(Y,predict_Y))
#Plotting



cmap = ListedColormap(['blue', 'red'])                    

plt.scatter(X1,X2, c=predict_Y,marker='.', cmap=cmap)

#plt.scatter(errors,errors, c=errors, marker='*',s=100,edgecolors="black",facecolors="none")

plt.show()
def do_fit_logreg(classifier, X, Y):

    classifier.fit(X, Y)

    predict_y1 = classifier.predict(X=X)

    success = 1-sum(abs(predict_y1 - Y))/len(Y)

    print("Hypothesis prediction success rate is %.2f" %success)

    print(classifier.score(X,Y))

    print(accuracy_score(Y,predict_y1))

    

    cmap = ListedColormap(['blue', 'red'])                    

    plt.scatter(X1,X2, c=predict_y1,marker='.', cmap=cmap)

    plt.show()
from sklearn.linear_model import LogisticRegression

logregressor = LogisticRegression(solver="liblinear")

do_fit_logreg(logregressor, X_NEW, Y) 
from matplotlib.colors import ListedColormap

myColorMap = ListedColormap(['blue', 'red'])                    



plt.figure(figsize=(8,6))

# Set min and max values and give it some padding

x1_min, x1_max = X[:,0].min() - 1, X[ :,0].max() + 1

x2_min, x2_max = X[:,1].min() - 1, X[ :,1].max() + 1



xx1 =np.linspace(x1_min, x1_max, 50)

xx2 =np.linspace(x2_min, x2_max, 50)

#Plot Prediction Data

for i in range(len(xx1)):

    for j in range(len(xx2)):

        newX=(np.column_stack((xx1[i],xx2[j])))

        #newY=predict(newX, weights) 

        

        

        newY = clf.predict(PolyFeatures.fit_transform(scaler.transform(newX)))

        yColor=myColorMap(int(newY))

        plt.scatter(xx1[i].flatten(),xx2[j].flatten(),color=yColor,alpha=0.4);



#Plot Training data

plt.scatter(X[:,0].flatten(),X[:,1].flatten(), c=Y.flatten(),  cmap=myColorMap,edgecolor='k');

plt.show()