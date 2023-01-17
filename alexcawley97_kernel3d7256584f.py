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

import pandas as pd 

from sklearn.ensemble import BaggingClassifier

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import numpy as np

from matplotlib.ticker import MaxNLocator

from matplotlib import cm



# Open data csvshttps://reggienet.illinoisstate.edu/access/content/attachment/33176445-5f8e-4741-96e8-9f24f04aa1a2/Assignments/b727bd20-80f9-4614-afcc-40b18e27f99d/logReg-p1.jpg

titanicTrain = pd.read_csv("../input/titanic/cleanTitanicTrain.csv")

titanicTest = pd.read_csv("../input/titanic/cleanTitanicTest.csv")



# Columns used to predict

predictorNames = ["Pclass", 

                  "Sex", 

                  "Age", 

                  "SibSp", 

                  "Parch", 

                  "Fare", 

                  "Embarked"]



# Create the predictors and targets

trainPredictors = titanicTrain[predictorNames]

targetName = ["Survived"]

trainTarget = titanicTrain["Survived"]

testPredictors = titanicTest[predictorNames]



# Create teh logistic regressor

logReg = LogisticRegression(tol = 0.0000000001,

                            warm_start = True,

                            penalty = 'l2',

                            solver = 'lbfgs',

                            max_iter = 50000)



#Initialize bagging scores data frame

bagging_scores = np.zeros((10,7))



# Run bagging on logistic regressor

for s in range(0, 10):

    samples = 1 / (s + 1)

    for f in range (0, 7):

        features = 1 / (f + 1)

        bagging = BaggingClassifier(logReg,

                                    max_samples = samples,

                                    max_features = features)

        bagging.fit(trainPredictors,

                    trainTarget)

        bagging_scores[s, f] = round(bagging.score(trainPredictors,

                                                   trainTarget) * 100,

                                                   2)

        

# Create the figure

fig = plt.figure()

ax = fig.add_subplot(111,

                     projection = "3d")

x = np.arange(1,

              10,

              1)

y = np.arange(1,

              7,

              1)

x, y = np.meshgrid(x,

                   y)

z = np.array(bagging_scores[x, 

                            y])

    

# Create the surface of the plot

surf = ax.plot_surface(x,

                       y,

                       z,

                       rstride = 1,

                       cstride = 1,

                       cmap = cm.jet,

                       linewidth = 0)

fig.colorbar(surf)

title = ax.set_title("Log reg scores, given two parameters of bagging classifier")

title.set_y(1.01)

ax.xaxis.set_major_locator(MaxNLocator(5))

ax.yaxis.set_major_locator(MaxNLocator(6))

ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

fig.savefig("3D-constructing-{}.png".format(1))

        