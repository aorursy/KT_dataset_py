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
heart = pd.read_csv("../input/heart-disease-uci/heart.csv")
heart.shape
heart.head()
heart.describe()
heart.info()
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics
x = heart.iloc[:,0:13]

x
#Getting the feature_names

col_names = heart.drop('target',axis = 1).columns

print(col_names)
y = heart.loc[:,"target"]

y
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)
from sklearn.linear_model import LogisticRegression #import Logistic Regression
logReg = LogisticRegression()    #make instance of Logistic Regression
logReg.fit(x_train,y_train)    #train model
predictions = logReg.predict(x_test)    #using model to predict the y values given the x values
# Making the Confusion Matrix    #check the performance of model against actual y values

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

cm
#Other ways of accessing performance

accuracyLogReg = metrics.accuracy_score(y_test,predictions)    #accuracy score

accuracyLogReg
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
#Each of the coefficients correspond to their feature_names. 

for i in range(len(logReg.coef_[0])):    

    print("Coeeficient of " + col_names[i] + " : " + str(logReg.coef_[0][i]))
# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(x_train, y_train)
from sklearn import tree

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [20, 10]

tree.plot_tree(classifier,fontsize = 10)
# Predicting the Test set results

y_pred = classifier.predict(x_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
print(classification_report(y_test,y_pred))
#Attempt to reduce max_depth of tree to see if improve classification accuracy and interpretability 



from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = 5)

classifier.fit(x_train, y_train)



#Change plot size

plt.rcParams['figure.figsize'] = [20, 10]



tree.plot_tree(classifier,fontsize = 10,feature_names = col_names)
# Predicting using 2nd tree

y_pred = classifier.predict(x_test)

print(classification_report(y_test,y_pred))
import matplotlib.pyplot as plt

plt.bar(x = col_names, height = classifier.feature_importances_)

plt.show()
array = classifier.feature_importances_
import numpy as np

index = array.argsort()[-3:][::-1]

for i in index:

    print(col_names[i])
#Attempt to max_leaf_nodes of tree to see if improve classification accuracy and interpretability 



from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_leaf_nodes = 10)

classifier.fit(x_train, y_train)



#Change plot size

plt.rcParams['figure.figsize'] = [20, 10]



tree.plot_tree(classifier,fontsize = 10,feature_names = col_names)
# Predicting using 3rd tree

y_pred = classifier.predict(x_test)

print(classification_report(y_test,y_pred))
import matplotlib.pyplot as plt

plt.bar(x = col_names, height = classifier.feature_importances_)

plt.show()
array = classifier.feature_importances_

index = array.argsort()[-3:][::-1]

for i in index:

    print(col_names[i])