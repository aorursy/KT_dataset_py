# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
# read in data

df = pd.read_csv('../input/Iris.csv')
# preview dataset

df.head()
# numerical statistics of dataset

df.describe()
# view the relationships between variables; color code by species type

sns.pairplot(df.drop(labels=['Id'], axis=1), hue='Species')
# split data into training and test sets; set random state to 0 for reproducibility 

X_train, X_test, y_train, y_test = train_test_split(df[['SepalLengthCm', 'SepalWidthCm', 

                                                        'PetalLengthCm', 'PetalWidthCm']],

                                                    df['Species'], random_state=0)
# see how data has been split

print("X_train shape: {}\ny_train shape: {}".format(X_train.shape, y_train.shape))

print("X_test shape: {}\ny_test shape: {}".format(X_test.shape, y_test.shape))
# initialize the Estimator object

knn = KNeighborsClassifier(n_neighbors=1)
# fit the model to training set in order to predict classes

knn.fit(X_train, y_train)
# create a prediction array for our test set

y_pred = knn.predict(X_test)
# based on the training dataset, our model predicts the following for the test set:

pd.concat([X_test, y_test, pd.Series(y_pred, name='Predicted', index=X_test.index)], 

          ignore_index=False, axis=1)
# what is our score?

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))