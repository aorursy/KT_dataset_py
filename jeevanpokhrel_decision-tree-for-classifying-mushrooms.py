# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Importing the dataset

dataset1 = pd.read_csv('../input/mushrooms.csv')
# Visualising the dataset

dataset1.head(5)
# Information about the tables

dataset1.info()
# Find the number of unique values  in each feature

features = dataset1.drop('class',axis=1)

for feature in features:

    print('%s has %d unique values' % (feature, len(np.unique(dataset1[feature]))))
#Randomly selected 3 featured habitat, odor, populational for classification

dataset2 = dataset1[['class','habitat','odor','population']]
from sklearn.preprocessing import LabelEncoder

#  label encoding to all the columns 

for col in dataset2.columns.values:

    dataset3[col] = LabelEncoder().fit_transform(dataset2[col])

dataset3.head() 
sns.countplot(x = dataset3['class'], data=dataset3)
from sklearn.model_selection import train_test_split
#Modelling with Decision tree

X = dataset3.drop('class',axis=1)

y = dataset3['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
#Prediction accuracy

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
#Model evaluation with ROC

from sklearn.metrics import roc_curve

y_pred_prob = dtree.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--') 

plt.plot(fpr, tpr, label='Logistic Regression') 

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate') 

plt.title('Logistic Regression ROC Curve')

plt.show()

    