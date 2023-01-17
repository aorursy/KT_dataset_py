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
from __future__ import print_function

import os

import warnings

import numpy as np

import matplotlib.pyplot as plt

import pandas_profiling

import matplotlib.image as image

import pandas as pd
plt.style.use("ggplot")

warnings.simplefilter("ignore")

plt.rcParams['figure.figsize'] = (12,8)
data = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data_original = data

data.head()
data.profile_report(title = "Data")
pd.crosstab(data.BusinessTravel,data.Attrition).plot(kind = "bar")

plt.title('Turnover Frequency on Business Travel Bracket')

plt.xlabel('Travel')

plt.ylabel('Frequency of Turnover')

plt.show()

pd.crosstab(data.Department,data.Attrition).plot(kind = "bar")

plt.title('Turnover Frequency on Department Bracket')

plt.xlabel('Department')

plt.ylabel('Frequency of Turnover')

plt.show()

pd.crosstab(data.EducationField,data.Attrition).plot(kind = "bar")

plt.title('Turnover Frequency on Edication Field Bracket')

plt.xlabel('Education Field')

plt.ylabel('Frequency of Turnover')

plt.show()

pd.crosstab(data.Gender,data.Attrition).plot(kind = "bar")

plt.title('Turnover Frequency on Gender')

plt.xlabel('Gender')

plt.ylabel('Frequency of Turnover')

plt.show()
data.info()
cat_vars = ['EducationField','Department','BusinessTravel','Gender','JobRole','MaritalStatus','Over18','OverTime']

for i in cat_vars:

    cat_list = 'var'+'_'+i

    cat_list = pd.get_dummies(data[i],prefix = i)

    hr = data.join(cat_list)

    data = hr

    
data.drop(columns=['EducationField','Department','BusinessTravel','Gender','JobRole','MaritalStatus','Over18','OverTime'],axis=1,inplace= True)
data.columns
data.head()
data.Attrition = data.Attrition.replace(to_replace=['No','Yes'], value=[0,1])
data.head()
X = data.loc[:,data.columns != 'Attrition']

y = data.Attrition
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify = y)
X_train.head()
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook

from IPython.display import SVG

from graphviz import Source

from IPython.display import display

from ipywidgets import interactive, IntSlider, FloatSlider, interact

import ipywidgets

from IPython.display import Image

from subprocess import call

import matplotlib.image as mpimg
@interact



def plot_DecisionTree(crit=["gini","entropy"],

             split = ["best","random"],

             depth = IntSlider(min = 1,max=30,value=2,continuous_update = False),

             min_split = IntSlider(min = 2, max=5, value=2,continuous_update = False),

             min_leaf = IntSlider(min=1, max=5,value=1,continuous_update = False)):

    

    estimator = DecisionTreeClassifier(random_state=0,

                                     criterion=crit,

                                     splitter = split,

                                     max_depth = depth,

                                     min_samples_split=min_split,

                                     min_samples_leaf=min_leaf)

    

    estimator.fit(X_train,y_train)

    print('Decision Tree Training Accuracy: {:.3f}'.format(accuracy_score(y_train, estimator.predict(X_train))))

    print('Decision Tree Test Accuracy: {:.3f}'.format(accuracy_score(y_test, estimator.predict(X_test))))

    

    graph = Source(tree.export_graphviz(estimator,

                                        out_file=None,

                                        feature_names=X_train.columns,

                                        class_names=['0', '1'],

                                        filled = True))

    

    display(Image(data=graph.pipe(format='png')))

    

    return estimator



    
@interact 



def polt_RandomForestTree(crit=["gini", "entropy"],

                         bootstrap=["True", "False"],

                         depth=IntSlider(min=1,max=30,value=3, continuous_update=False),

                         forests=IntSlider(min=1,max=200,value=100,continuous_update=False),

                         min_split=IntSlider(min=2,max=5,value=2, continuous_update=False),

                         min_leaf=IntSlider(min=1,max=5,value=1, continuous_update=False)):

    

    estimator = RandomForestClassifier(random_state=1,

                                       criterion=crit,

                                       bootstrap=bootstrap,

                                       n_estimators=forests,

                                       max_depth=depth,

                                       min_samples_split=min_split,

                                       min_samples_leaf=min_leaf,

                                       n_jobs=-1,

                                      verbose=False).fit(X_train, y_train)

    

    print('Random Forest Training Accuracy: {:.3f}'.format(accuracy_score(y_train, estimator.predict(X_train))))

    print('Random Forest Test Accuracy: {:.3f}'.format(accuracy_score(y_test, estimator.predict(X_test))))

    num_tree = estimator.estimators_[0]

    print('\nVisualizing Decision Tree:', 0)

    

    graph = Source(tree.export_graphviz(num_tree,

                                        out_file=None,

                                        feature_names=X_train.columns,

                                        class_names=['0', '1'],

                                        filled = True))

    

    display(Image(data=graph.pipe(format='png')))

    

    return estimator
from yellowbrick.model_selection import FeatureImportances

plt.rcParams['figure.figsize'] = (12,8)

plt.style.use("ggplot")



rf = RandomForestClassifier(bootstrap='True', class_weight=None, criterion='gini',

            max_depth=3, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,

            oob_score=False, random_state=1, verbose=False,

            warm_start=False)



viz = FeatureImportances(rf)

viz.fit(X_train, y_train)

viz.show();
from yellowbrick.classifier import ROCAUC



visualizer = ROCAUC(rf, classes=["Yes", "No"])



visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.poof();
dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,

            max_features=None, max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, presort=False, random_state=0,

            splitter='best')



visualizer = ROCAUC(dt, classes=["No", "Yes"])



visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.poof();
from sklearn.linear_model import LogisticRegressionCV



logit = LogisticRegressionCV(random_state=1, n_jobs=-1,max_iter=500,

                             cv=10)



lr = logit.fit(X_train, y_train)



print('Logistic Regression Accuracy: {:.3f}'.format(accuracy_score(y_test, lr.predict(X_test))))



visualizer = ROCAUC(lr, classes=["Yes", "No"])



visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.poof();