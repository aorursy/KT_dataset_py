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

student_job = pd.read_csv("../input/student-job/student_job.csv")

student_job.head()
from sklearn import preprocessing



data = student_job.apply(preprocessing.LabelEncoder().fit_transform)

data.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB



predictors = data.iloc[:,0:4] # Segregating the predictor variables ...

target = data.iloc[:,4] # Segregating the target / class variable ...

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size = 0.3, random_state = 123)

gnb = GaussianNB()

# First train model / classifier with the input dataset (training data part of it)

model = gnb.fit(predictors_train, target_train)

# Make prediction using the trained model

prediction = model.predict(predictors_test)

# Time to check the prediction accuracy ...

accuracy_score(target_test, prediction, normalize = True)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier



predictors = data.iloc[:,0:4] # Segregating the predictors..

target = data.iloc[:,4] # Segregating the target variable ...

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size = 0.3, random_state = 123)

dtree_entropy = DecisionTreeClassifier(random_state = 100, max_depth=3)

model = dtree_entropy.fit(predictors_train, target_train)

prediction = dtree_entropy.predict(predictors_test)

acc_score = accuracy_score(target_test, prediction, normalize = True)

print(acc_score)
from IPython.display import Image  

from sklearn import tree

from sklearn.externals.six import StringIO  

import pydot          



dot_data = StringIO() 

tree.export_graphviz(dtree_entropy, out_file=dot_data)  

graph = pydot.graph_from_dot_data(dot_data.getvalue()) 

graph[0].write_pdf("Dtree.pdf")