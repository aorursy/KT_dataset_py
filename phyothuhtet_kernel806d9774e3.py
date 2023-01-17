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
import pandas as pd

from sklearn import datasets



iris = datasets.load_iris()

species = [iris.target_names[x] for x in iris.target]

iris  = pd.DataFrame(iris['data'],columns =['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'])

iris['Species'] = species



print(iris)
iris['count'] = 1

iris[['Species','count']].groupby('Species').count()

print(iris)
            

import matplotlib.pyplot as plt

%matplotlib inline



            
def plot_iris():

    setosa = iris[iris['Species']=='setosa']

    versicolor = iris[iris['Species']=='versicolor']

    virginica = iris[iris['Species']=='virginica']

    

    fig,ax = plt.subplots(2,2,figsize=(12,12))

    x     =['Sepal_Length','Sepal_Width']

    y     =['Petal_Length','Petal_Width']

    

    for i in range(2):

        for j in range(2):

            ax[i][j].scatter(setosa[x[i]],setosa[y[j]])

            ax[i][j].scatter(versicolor[x[i]],versicolor[y[j]])

            ax[i][j].scatter(virginica[x[i]],virginica[y[j]])

            ax[i][j].set_xlabel(x[i])

            ax[i][j].set_ylabel(y[i])
plot_iris()
import numpy as np

Features = np.array(iris[['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']])

levels = {'setosa':0,'versicolor':1,'virginica':2}

Labels = np.array([levels[x] for x in iris['Species']])
import sklearn.model_selection as ms



indx = range(Features.shape[0])

indx = ms.train_test_split(indx,test_size = 100)



X_train = Features[indx[0],:]

y_train = np.ravel(Labels[indx[0]])



X_test = Features[indx[1],:]

y_test = np.ravel(Labels[indx[1]])



import sklearn.preprocessing as preprocessing



scale = preprocessing.StandardScaler()

scale.fit(X_train)

X_train = scale.transform(X_train)

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes = (50,))

model.fit(X_train,y_train)
import numpy.random as nr

nr.seed(123)

X_test = scale.transform(X_test)

scores = model.predict(X_test)
import sklearn.metrics as sklm

def print_metrics(labels,score):

    conf = sklm.confusion_matrix(labels,score)

    print("Score Setosa","Score Versicolor","Score Virginica")

    print("Actual Setosa",conf[0,0],conf[0,1],conf[0,2])

    print("Actual Versicolor",conf[1,0],conf[1,1],conf[1,2])

    print("Actual Virginica",conf[2,0],conf[2,1],conf[2,2])

    print("Accuracy",sklm.accuracy_score(labels,score))

    m = sklm.precision_recall_fscore_support(labels,score)

    print('numcase',m[3][0],m[3][1],m[3][2])

    print('setosa',m[1][0],m[1][1],m[1][2])

    print('versicolor',m[2][0],m[2][1],m[2][2])

    

print_metrics(y_test,scores)   
def plot_iris_score(iris, y_test, scores):

    '''Function to plot iris data by type'''

    ## Find correctly and incorrectly classified cases

    true = np.equal(scores, y_test).astype(int)

    print(true)



    

    ## Create data frame from the test data

    iris = pd.DataFrame(iris)

    levels = {0:'setosa', 1:'versicolor', 2:'virginica'}

    iris['Species'] = [levels[x] for x in y_test]

    iris.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']

    

    ## Set up for the plot

    fig, ax = plt.subplots(2, 2, figsize=(12,12))

    markers = ['o', '+']

    x_ax = ['Sepal_Length', 'Sepal_Width']

    y_ax = ['Petal_Length', 'Petal_Width']

    

    for t in range(2): # loop over correct and incorect classifications

        setosa = iris[(iris['Species'] == 'setosa') & (true == t)]

        versicolor = iris[(iris['Species'] == 'versicolor') & (true == t)]

        virginica = iris[(iris['Species'] == 'virginica') & (true == t)]

        # loop over all the dimensions

        for i in range(2):

            for j in range(2):

                ax[i,j].scatter(setosa[x_ax[i]], setosa[y_ax[j]], marker = markers[t], color = 'blue')

                ax[i,j].scatter(versicolor[x_ax[i]], versicolor[y_ax[j]], marker = markers[t], color = 'orange')

                ax[i,j].scatter(virginica[x_ax[i]], virginica[y_ax[j]], marker = markers[t], color = 'green')

                ax[i,j].set_xlabel(x_ax[i])

                ax[i,j].set_ylabel(y_ax[j])



plot_iris_score(X_test, y_test, scores)