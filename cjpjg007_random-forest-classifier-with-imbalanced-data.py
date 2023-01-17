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
train=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

train.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,f1_score

from sklearn.model_selection import train_test_split
x=train.iloc[:20000,:-1]

y=train.iloc[:20000,-1]
y
(x_train,x_test,y_train,y_test)=train_test_split(x,y,random_state=0)

x_train.shape,x_test.shape,y_train.shape,y_test.shape
classifier=RandomForestClassifier()

classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

print("Test Accuracy Score:{}".format(accuracy_score(y_pred,y_test)))



print("Test F1 Score:{}".format(f1_score(y_pred,y_test)))



print("Train Accuracy Score:{}".format(accuracy_score(classifier.predict(x_train),y_train)))



print("Train F1 Score:{}".format(f1_score(classifier.predict(x_train),y_train)))