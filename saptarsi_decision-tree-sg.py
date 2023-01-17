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
! pip install pydotplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
df=pd.read_csv("/kaggle/input/uciseeds/seeds.csv")
print(df.shape)
print(df.describe())
df.head(2)
X=df.iloc[:,0:7]
y=df.iloc[:,7]
feature_names=list(X.columns)
print(feature_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test
clf = DecisionTreeClassifier()
#clf = DecisionTreeClassifier(min_samples_leaf = 20)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_names,class_names=['1','2','3'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('seeds.png')
Image(graph.create_png())
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
print('\nConfusion matrix')
print(confusion_matrix(y_test, y_pred))

#Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report')
print(classification_report(y_test, y_pred))  
nc=np.arange(1,50,2)
acc=np.empty(25)
i=0
for k in np.nditer(nc):
    clf = DecisionTreeClassifier(max_depth = int(k))
    clf.fit(X_train, y_train)
    temp= clf.score(X_test, y_test)
    acc[i]=temp
    i = i + 1
x=pd.Series(acc,index=nc)
x.plot()
# Add title and axis names
plt.title('Max Depth vs Accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show() 
nc=np.arange(1,1000,2)
acc=np.empty(500)
i=0
for k in np.nditer(nc):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(k))
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    temp= clf.score(X_test, y_test)
    acc[i]=temp
    i = i + 1
x=pd.Series(acc,index=nc)
x.plot(kind='box')
# Add title and axis names
plt.title('Accuracy on diffrent test set')
plt.xlabel('Seed valus')
plt.ylabel('Accuracy')
plt.show() 
# Reading the File
df=pd.read_csv("/kaggle/input/advtlr/Advertising.csv")
# Inspecting the dataset
df.shape
df.head(2)
#Importing the decsion tree regressor
from sklearn.tree import DecisionTreeRegressor
#Setting up X and y
X=df.iloc[:,1:4]
y=df.iloc[:,4]
regressor = DecisionTreeRegressor(random_state=0)
# Splitting on train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=408)
#Test the model
regressor.fit(X_train,y_train)
# Check R Squared
print(regressor.score(X_test,y_test))