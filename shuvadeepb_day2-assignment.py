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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
mydata = pd.read_csv("../input/breastcancer-dataset/data.csv")
print(mydata.shape)
print(mydata.head())

#remove 2 columns
mydata=mydata.drop(['id', 'Unnamed: 32'], axis = 1) 
print(mydata.shape)
#set input variable
y = mydata.iloc[:,0]
x = mydata.iloc[:,1:31]
#data normalisation
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x)
x_scaled
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)
print('Accuracy = ', knn.score(x_test, y_test))
knn1=KNeighborsClassifier(n_neighbors=15,weights='distance')
knn1.fit(x_train,y_train)
y1_pred=knn1.predict(x_test)
print('Accuracy=',knn1.score(x_test,y_test))