import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.info()
sns.set_style("dark")

fig = plt.figure(figsize = [15,15])

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

i = 1

for col in cols :

    plt.subplot(4,3,i)

    sns.barplot(data = df, x = 'quality', y = col)

    i=i+1

plt.show()  
sns.pairplot(df)

plt.show()
x = df.drop('quality',1)

y = df['quality']
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=40)
tree = DecisionTreeClassifier()

tree.fit(x_train, y_train)
y_pred= tree.predict(x_test)
y_pred
metrics.accuracy_score(y_test,y_pred)
grid_param = {

    'max_depth' : range(4,20,4),

    'min_samples_leaf' : range(20,200,40),

    'min_samples_split' : range(20,200,40),

    'criterion' : ['gini','entropy'] 

}
d_tree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=d_tree,

                     param_grid=grid_param,

                     cv=5,

                    n_jobs =-1)
grid_search.fit(x_train,y_train)
best_parameters = grid_search.best_params_

print(best_parameters)
grid_search.best_score_
clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, min_samples_split=60)

clf.fit(x_train,y_train)
clf.score(x_test,y_test)