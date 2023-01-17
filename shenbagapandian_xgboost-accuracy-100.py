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
# read data from csv

import pandas as pd 

df = pd.read_csv("../input/amd-vs-intel/AMDvIntel.csv")

df.head()
X = df.iloc[:,1:4] 

X.head()
y =df.IorA

y.head()
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from xgboost import XGBClassifier

model = XGBClassifier(max_depth =5,

                     n_estimators=100,

                     subsample=.8,

                     learning_rate=0.1,

                     reg_alpha=0,

                     reg_lambda=1,

                     colsample_bynode=0.6,

                     colsample_bytree=0.5,

                     gamma = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score



accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)

cm