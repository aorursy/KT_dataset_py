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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
df.head()
df.info()
X = df.drop(['class'], axis=1)

Y = df['class']
def encode(df, features):

    for ftr in features:

        df = pd.get_dummies(df, ftr)

    return df
X = encode(X,X)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"accuracy score is {accuracy_score(y_pred, y_test)}")
y_pred=pd.DataFrame({'index':X_test.index,'class':y_pred})

y_pred.head()

y_pred.to_csv('submission.csv',index=False)