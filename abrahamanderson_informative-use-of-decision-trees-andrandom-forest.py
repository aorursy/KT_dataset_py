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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv("../input/kyphosis.csv")

df.head()

df.info()

#There are 81 rows and 4 columns in the dataset
sns.pairplot(df,hue="Kyphosis")
from sklearn.model_selection import train_test_split

X=df.drop("Kyphosis",axis=1) # All of the columns except from the target column

y=df["Kyphosis"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier() # Here we create an instant of ht ealgorithm

dtree.fit(X_train,y_train) # here we make algorithm fit the training dataset
predictions=dtree.predict(X_test)

predictions
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,predictions))

print("\n")

print(confusion_matrix(y_test,predictions))
from sklearn import metrics
# Create Decision Tree classifer object

classifier = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer

classifier = classifier.fit(X_train,y_train)

#Predict the response for test dataset

y_pred = classifier.predict(X_test)

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)
rfc_predictions=rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_predictions))

print("\n")

print(classification_report(y_test,rfc_predictions))
print(classification_report(y_test,predictions))

print("\n")

print(confusion_matrix(y_test,predictions))

print(3*"\n")

print(confusion_matrix(y_test,rfc_predictions))

print("\n")

print(classification_report(y_test,rfc_predictions))