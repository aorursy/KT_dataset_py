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
from sklearn.model_selection import train_test_split,cross_val_score

from xgboost import XGBClassifier



path = '../input/user-id-class/Dataset.csv'

Dataset = pd.read_csv(path, header=None)

X, y = Dataset.iloc[: , 1:], Dataset.iloc[:, 0]

X_train,X_test, y_train,y_test = train_test_split (X,y,test_size=0.3)

eval_set = [(X_test, y_test)]

model = XGBClassifier(n_estimators = 100, learning_rate = 0.05)

model.fit(X_train,y_train,early_stopping_rounds=20, eval_metric="merror", eval_set=eval_set, verbose=True)

y_pred = model.predict(X_test)

val_scores = cross_val_score(model,X,y,cv=5)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

val_scores
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test, y_pred)

mat