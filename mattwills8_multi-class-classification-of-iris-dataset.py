# data imports
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# plot imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
from sklearn import linear_model
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
Y = iris.target
print(iris.DESCR)
# create dataframes for visualisations

iris_data = DataFrame(X, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
iris_target = DataFrame(Y, columns=['Species'])
# at the moment we have 0, 1 and2 for species, so we want to change that to make it clearer

def flower(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolour'
    else:
        return 'Virginica'
# label flowers# combine dataframes

iris_target['Species'] = iris_target['Species'].apply(flower)
# combine dataframes

iris = pd.concat([iris_data, iris_target], axis=1)
sns.pairplot(iris, hue='Species', size=2)
sns.factorplot('Petal Length', data=iris, hue='Species', size=5, kind='count')
# train model

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

log_reg = LogisticRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.4, random_state=3)

log_reg.fit(X_train, Y_train)
# test accuracy

from sklearn import metrics

Y_pred = log_reg.predict(X_test)

print(metrics.accuracy_score(Y_test, Y_pred))

