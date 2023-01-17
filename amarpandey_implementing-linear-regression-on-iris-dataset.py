# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualisation and plotting

import matplotlib.pyplot as plt # data plotting

import warnings



# Seaborn default configuration

sns.set_style("darkgrid")



# set the custom size for my graphs

sns.set(rc={'figure.figsize':(8.7,6.27)})



# filter all warnings

warnings.filterwarnings('ignore') 



# set max column to 999 for displaying in pandas

pd.options.display.max_columns=999 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/Iris.csv')
data.head()
data.info()
data.describe()
data['Species'].value_counts()
rows, col = data.shape

print("Rows : %s, column : %s" % (rows, col))
snsdata = data.drop(['Id'], axis=1)

g = sns.pairplot(snsdata, hue='Species', markers='x')

g = g.map_upper(plt.scatter)

g = g.map_lower(sns.kdeplot)
sns.violinplot(x='SepalLengthCm', y='Species', data=data, inner='stick', palette='autumn')

plt.show()

sns.violinplot(x='SepalWidthCm', y='Species', data=data, inner='stick', palette='autumn')

plt.show()

sns.violinplot(x='PetalLengthCm', y='Species', data=data, inner='stick', palette='autumn')

plt.show()

sns.violinplot(x='PetalWidthCm', y='Species', data=data, inner='stick', palette='autumn')

plt.show()
mapping = {

    'Iris-setosa' : 1,

    'Iris-versicolor' : 2,

    'Iris-virginica' : 3

}



X = data.drop(['Id', 'Species'], axis=1).values # Input Feature Values

y = data.Species.replace(mapping).values.reshape(rows,1) # Output values



X = np.hstack(((np.ones((rows,1))), X))# Adding one more column for bias
np.random.seed(0) # Let's set the zero for time being

theta = np.random.randn(1,5) # Setting values of theta randomly



print("Theta : %s" % (theta))
iteration = 10000

learning_rate = 0.003 # If you are going by formula, this is actually alpha.

J = np.zeros(iteration) # 1 x 10000 maxtix
# Let's train our model to compute values of theta

for i in range(iteration):

    J[i] = (1/(2 * rows) * np.sum((np.dot(X, theta.T) - y) ** 2 ))

    theta -= ((learning_rate/rows) * np.dot((np.dot(X, theta.T) - y).reshape(1,rows), X))



prediction = np.round(np.dot(X, theta.T))



ax = plt.subplot(111)

ax.plot(np.arange(iteration), J)

ax.set_ylim([0,0.15])

plt.ylabel("Cost Values", color="Green")

plt.xlabel("No. of Iterations", color="Green")

plt.title("Mean Squared Error vs Iterations")

plt.show()
ax = sns.lineplot(x=np.arange(iteration), y=J)

plt.show()
ax = plt.subplot(111)



ax.plot(np.arange(1, 151, 1), y, label='Orignal value', color='red')

ax.scatter(np.arange(1, 151, 1), prediction, label='Predicted Value')



plt.xlabel("Dataset size", color="Green")

plt.ylabel("Iris Flower (1-3)", color="Green")

plt.title("Iris Flower (Iris-setosa = 1, Iris-versicolor = 2, Iris-virginica = 3)")



ax.legend()

plt.show()
accuracy = (sum(prediction == y)/float(len(y)) * 100)[0]

print("The model predicted values of Iris dataset with an overall accuracy of %s" % (accuracy))