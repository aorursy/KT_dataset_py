import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn

from sklearn.preprocessing import LabelEncoder



plt.style.use('ggplot')

iris = pd.read_csv('../input/Iris.csv')

%matplotlib inline

iris.head()
from sklearn.cross_validation import train_test_split

X = iris[[c for c in iris.columns if c != "Species" and c!='Id']]

Y = iris["Species"]

Y = LabelEncoder().fit_transform(Y)

Id = iris['Id']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.50, train_size=.50, random_state=42)

type(x_train), type(y_train)
#Use pairplot to analyze the relationship between species for all characteristic combinations. 

# An observable trend shows a close relationship between two of the species



seaborn.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)

plt.show()
seaborn.pairplot(iris, hue="Species",diag_kind="kde",diag_kws=dict(shade=True))
iris.shape, y_test.shape, y_train.shape
iris[['Id','Species']].groupby("Species").count().reset_index()
#import pandas_ml

#from pandas_ml import ConfusionMatrix

#binary_confusion_matrix = ConfusionMatrix(y_train, y_test)

#binary_confusion_matrix.print_stats()



#The pandas_ml confusion matrix does not work with this kernal. 

#Please refer to the code on github for KNN classfication