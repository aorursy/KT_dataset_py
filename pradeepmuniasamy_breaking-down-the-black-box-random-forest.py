#EDA for the data

import pandas_profiling as pp

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder #label encoding the features

from sklearn.model_selection import train_test_split #Splitting the data as train and test

from sklearn.ensemble import RandomForestClassifier #Random Forrest

from sklearn import metrics #accuracy calculation



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/iris/Iris.csv")
report = pp.ProfileReport(data)

report
data["Species"]=LabelEncoder().fit_transform(data["Species"].astype(str)) 
X=data[["PetalLengthCm","PetalWidthCm","SepalLengthCm","SepalWidthCm"]]  # Features

y=data['Species']  # Labels



# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.datasets import load_iris

iris = load_iris()



# Model (can also use single decision tree)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10)



# Train

model.fit(iris.data, iris.target)

# Extract single tree

estimator = model.estimators_[5]



from sklearn.tree import export_graphviz

# Export as dot file

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = iris.feature_names,

                class_names = iris.target_names,

                rounded = True, proportion = False, 

                precision = 2, filled = True)



# Convert to png using system command (requires Graphviz)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')