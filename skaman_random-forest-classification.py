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
#importing basic library

import numpy as np

import pandas as pd
#loading dataset 

data_frame= pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
#quick look

data_frame.head()
#preparing data 

feature=['sepal_length','sepal_width','petal_length','petal_width']

X=data_frame[feature]

y=data_frame.species
#dividing into train test set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#training algorithm

from sklearn.ensemble import RandomForestClassifier

classification = RandomForestClassifier(n_estimators=20)

classification.fit(X_train,y_train)

y_pred=classification.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#demo prediction

classification.predict([[1,1,1,1]])