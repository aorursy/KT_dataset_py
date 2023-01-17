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
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
X = df[['radius_mean','texture_mean','perimeter_mean','area_mean']]
y =df['diagnosis']
y= pd.Series(np.where(y.values=='M',1,0),y.index)
y.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)
from sklearn.svm import SVC

svclassifier = SVC(kernel='linear', C =2.4)

svclassifier.fit(X_train,y_train)
y_pred = svclassifier.predict(X_test)
y_pred
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))