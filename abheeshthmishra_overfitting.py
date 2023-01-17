# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from sklearn import datasets

from sklearn import manifold



%matplotlib inline
Data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
Data.head(5)
p = Data['quality'].unique()

print(p)
quality_mapping = {

    3:0,

    4:1,

    5:2,

    6:3,

    7:4,

    8:5

}



Data.loc[:,"quality"] = Data.quality.map(quality_mapping)
# use sample with frac=1 to shuffle the dataframe

# we reset the indices since they change after

# shuffling the dataframe

Data = Data.sample(frac = 1).reset_index(drop=True)

Data.head()
# top 1000 rows are selected

# for training

data_train = Data.head(1000)

# bottom 599 values are selected

# for testing/validation

data_test = Data.tail(599)
# import from scikit-learn

from sklearn import tree

from sklearn import metrics

# initialize decision tree classifier class

# with a max_depth of 3

clf = tree.DecisionTreeClassifier(max_depth=3)

# choose the columns you want to train on

# these are the features for the model

cols = ['fixed acidity',

 'volatile acidity',

 'citric acid',

 'residual sugar',

 'chlorides',

 'free sulfur dioxide',

 'total sulfur dioxide',

 'density',

 'pH',

 'sulphates',

 'alcohol']

  

# train the model on the provided features

# and mapped quality from before

clf.fit(data_train[cols],data_train.quality)

    

# generate predictions on the training set

train_predictions = clf.predict(data_train[cols])



# generate predictions on the test set

test_predictions = clf.predict(data_test[cols])



# calculate the accuracy of predictions on

# training data set



training_accuracy = metrics.accuracy_score(

data_train.quality,train_predictions

)



# calculate the accuracy of predictions on

# test data set

test_accuracy = metrics.accuracy_score(

data_test.quality, test_predictions

)
print(test_accuracy,training_accuracy)
# import scikit-learn tree and metrics

from sklearn import tree

from sklearn import metrics

# import matplotlib and seaborn

# for plotting

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

# this is our global size of label text

# on the plots

matplotlib.rc('xtick', labelsize=20)

matplotlib.rc('ytick', labelsize=20)

# This line ensures that the plot is displayed

# inside the notebook

%matplotlib inline

# initialize lists to store accuracies

# for training and test data

# we start with 50% accuracy

train_accuracies = [0.5]

test_accuracies = [0.5]

# iterate over a few depth values

for depth in range(1, 25):

 # init the model

 clf = tree.DecisionTreeClassifier(max_depth=depth)

 # columns/features for training

 # note that, this can be done outside

 # the loop

 cols = [

 'fixed acidity',

 'volatile acidity',

 'citric acid',

 'residual sugar',

 'chlorides',

 'free sulfur dioxide',

 'total sulfur dioxide',

 'density',

 'pH',

 'sulphates',

 'alcohol'

 ]

 # fit the model on given features

 clf.fit(data_train[cols], data_train.quality)

 # create training & test predictions

 train_predictions = clf.predict(data_train[cols])

 test_predictions = clf.predict(data_test[cols])

 # calculate training & test accuracies

 train_accuracy = metrics.accuracy_score(

 data_train.quality, train_predictions

 )

 test_accuracy = metrics.accuracy_score(

 data_test.quality, test_predictions

 )



 # append accuracies

 train_accuracies.append(train_accuracy)

 test_accuracies.append(test_accuracy)

# create two plots using matplotlib

# and seaborn

plt.figure(figsize=(10, 5))

sns.set_style("whitegrid")

plt.plot(train_accuracies, label="train accuracy")

plt.plot(test_accuracies, label="test accuracy")

plt.legend(loc="upper left", prop={'size': 15})

plt.xticks(range(0, 26, 5))

plt.xlabel("max_depth", size=20)

plt.ylabel("accuracy", size=20)

plt.show()
