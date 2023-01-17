# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#********************************

import pandas as pd

import numpy as np

from sklearn.feature_selection import VarianceThreshold

from sklearn import svm

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.preprocessing import PowerTransformer

from sklearn.ensemble import BaggingClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_gaussian_quantiles

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_gaussian_quantiles

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.


def loadData (Xpaths):

    data = pd.read_csv(Xpaths) 

    return data

def preprocessing (Xtra):

    train_X = Xtra.iloc[:,:-1]

    train_y = Xtra.iloc[:,-1]

    #Scaler

    #scaler = StandardScaler()

    #scaler.fit(train_X)

    #train_X = scaler.transform(train_X)

    

    # PCA

    pca = PCA()

    pca.fit(train_X)

    train_X = pca.transform(train_X)

    print(train_X)

    return Xtra

def trainModel (Xtra):

    # Training classifiers

    classifier1 = DecisionTreeClassifier(max_depth=6,splitter='random',min_samples_leaf=3)

    classifier2 = KNeighborsClassifier(n_neighbors=7)

    classifier3 = SVC(kernel='linear', probability=True)

    classifier4 = AdaBoostClassifier(n_estimators=75,algorithm='SAMME.R')

    classifier5 = LogisticRegression(solver='lbfgs')

    model = VotingClassifier(estimators=[('dt', classifier1), ('knn', classifier2),

                                        ('svc', classifier3),('abdt', classifier4),('sd',classifier5),],

                            voting='soft',weights=[2,2,2,2,2])

    train_X = Xtra[Xtra.columns[:-1]]

    train_y = Xtra[Xtra.columns[-1]]  

    model.fit(train_X, train_y.ravel())

    return model

def predict(model,Xtst):

    predicted = model.predict(Xtst)

    predicted = np.around(predicted)

    predicted = predicted.astype(dtype=int)

    writeOutput(predicted)
def writeOutput(predicted):

    with open('submission.csv','w') as file:

        file.write('ID,Predicted\n')

        for i in range(len(predicted)):

            file.write('{},{}\n'.format(i+1,predicted[i]))

def result():

    Xpaths = "../input/train.csv"

    Xtra = loadData(Xpaths)

    Xpaths = "../input/test.csv"

    Xtst = loadData(Xpaths)

    Xtra = preprocessing (Xtra)

    model = trainModel (Xtra)

    predict(model,Xtst)



result()   