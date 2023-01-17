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
stevens = pd.read_csv("/kaggle/input/supremecourt-data/stevens.csv")

stevens.head()
X= stevens.iloc[:,2:8]

X.head()

Y = stevens.iloc[:,8]

Y.head()
from sklearn.preprocessing import LabelEncoder

lab = X.apply(LabelEncoder().fit_transform)

lab.head(30)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(lab, Y, test_size=0.25)
from sklearn.tree import DecisionTreeClassifier 

  

# create a regressor object 

regressor = DecisionTreeClassifier()  

  

# fit the regressor with X and Y data 

regressor.fit(X_train,y_train) 
y_pred = regressor.predict(X_test) 

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

df.head(20)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))