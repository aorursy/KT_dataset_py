# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris = datasets.load_iris()


print(iris.target_names)



print(iris.feature_names)
print(iris.data[0:5])
print(iris.target)
data=pd.DataFrame({

    'sepal length':iris.data[:,0],

    'sepal width':iris.data[:,1],

    'petal length':iris.data[:,2],

    'petal width':iris.data[:,3],

    'species':iris.target

})

data.head()
from sklearn.model_selection import train_test_split



X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features

y=data['species']  # Labels



# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
import matplotlib.pyplot as plt



plt.plot(y_test)
plt.plot(y_pred)
plt.plot(y_test, y_pred)
clf.predict([[3, 5, 4, 2]])
from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)
feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)

feature_imp
import seaborn as sns

%matplotlib inline

# Creating a bar plot

sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to your graph

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.legend()

plt.show()
from sklearn.model_selection import train_test_split

# Split dataset into features and labels

X=data[['petal length', 'petal width','sepal length']]  # Removed feature "sepal length"

y=data['species']                                       

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=5)
from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



# prediction on test set

y_pred=clf.predict(X_test)



#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
plt.plot(y_pred)
plt.plot(y_pred)
plt.plot(y_test, y_pred)