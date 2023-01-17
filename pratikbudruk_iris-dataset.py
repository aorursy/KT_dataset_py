# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Importing the dataset

dataset = pd.read_csv('../input/Iris.csv')

dataset.head()
#visualising the dataset

iris = data.drop('Id', axis=1)

g = sns.pairplot(iris, hue='Species', markers='+')

plt.show()
#dependent and independent variable

X=dataset.iloc[:,1:5].values

y=dataset.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
# Fitting Kernel SVM to the Training set

from sklearn.svm import SVC

classifier=SVC(kernel='rbf',random_state=0)

classifier.fit(X_train,y_train)
# Predicting the Test set results

y_pred = classifier.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
#accuracy

from sklearn.metrics import accuracy_score

acc=accuracy_score(y_pred,y_test)

acc