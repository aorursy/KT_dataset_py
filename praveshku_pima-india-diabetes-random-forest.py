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
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics



#Load Dataset

df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')



#Split X and y dataset

X = np.array(df.drop('Outcome', axis=1))

y = np.array(df['Outcome'])



#Split data into training set and testing set model_selection

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



#Feed data into Random Forest Algorithm

clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_train, y_train)



#Check accuracy using score method

accuracy = clf.score(X_test, y_test)

print('accuracy', accuracy)



#Test accuracy using matrics

y_pred = clf.predict(X_test)

print('Accuracy Score:', metrics.accuracy_score(y_test,y_pred))



#Check confusion metrics

print('confusion metrics :', metrics.confusion_matrix(y_test, y_pred))
