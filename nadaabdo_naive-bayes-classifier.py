import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix



sns.set()

%matplotlib inline

import os





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

iris = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')





print ("Number of instances ::", iris.shape[0])

print("Number of features ::", iris.shape[1])

print("Different species :: ",iris['species'].unique())

species_num = {'Iris-setosa' : 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3 }

iris['species'] = iris['species'].map(species_num)





iris.head()
sns.pairplot(iris, hue = "species")
corr = iris.corr()



fig, axis = plt.subplots(figsize=(15, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5)

#Here we split the date for training and testing

X=iris.drop('species',axis=1)

y=iris['species']

train_X,test_X,train_y,test_y=train_test_split(X,y)


GNB=GaussianNB()

GNB.fit(train_X,train_y)

pred_y=GNB.predict(test_X)



print("Accuracy : {}".format(GNB.score(test_X,test_y)))

mat=confusion_matrix(pred_y,test_y)



species = np.unique(pred_y)

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,

            xticklabels=species, yticklabels=species)

plt.xlabel('Truth')

plt.ylabel('Predicted')