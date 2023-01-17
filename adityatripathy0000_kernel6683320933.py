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
dataset = pd.read_csv('../input/parkinsons-data-set/parkinsons.data')

X = dataset.loc[:,dataset.columns!='status'].values[:,1:]

y = dataset.loc[:,'status'].values



#Splitting_Dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)



#Fitting XGBoost model to training set

from xgboost import XGBClassifier

classifier = XGBClassifier(n_estimators = 50, gamma = 0.1, max_depth = 5)

classifier.fit(X_train, y_train)



#predicting

y_pred = classifier.predict(X_test)



#accuracy

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)

print(acc*100)



#confusion_matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



#K-fold cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.mean()

accuracies.std()