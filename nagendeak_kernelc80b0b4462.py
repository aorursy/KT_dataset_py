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
import pandas as pd

import sklearn



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



training=pd.read_csv('../input/training.csv')

test=pd.read_csv('../input/testing.csv')



X_train=training.drop(['class'],axis=1)

y_train=training['class']

clf=LogisticRegression(random_state=42)



clf.fit(X_train,y_train)

X_test=test.drop(['class'],axis=1)

y_test=test['class']



y_pred=clf.predict(X_test)

print("Classification Report:")

print(sklearn.metrics.classification_report(y_test,y_pred))

print("Confustion Matrix")

print(sklearn.metrics.confusion_matrix(y_test,y_pred))

print("Accuracy Score: ",sklearn.metrics.accuracy_score(y_test,y_pred))