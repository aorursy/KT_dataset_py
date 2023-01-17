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
# For handling datasets
import pandas as pd

# For plotting graph
from matplotlib import pyplot as plt

# Import sklearn lib for KNN
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv('../input/iriscsv/Iris.csv')
df.head()
df.tail()
df.info()
df.columns
X=df.loc[:,'SepalLengthCm':'PetalWidthCm']
Y=df.loc[:,'Species']
knn=KNeighborsClassifier()

# Train the Model
knn.fit(X,Y)
X_test=[[4.9,7.0,1.2,0.2],
        [6.0,2.9,4.5,1.5],
        [6.1,2.6,5.6,1.2]]

# Test the model
prediction=knn.predict(X_test)
print(prediction)
# plot each relation 
# feature with each class

plt.xlabel('Feature')
plt.ylabel('Species')

X=df.loc[:,'SepalLengthCm']
Y=df.loc[:,'Species']
plt.scatter(X,Y,color='green',label='SepalLengthCm')


X=df.loc[:,'SepalWidthCm']
Y=df.loc[:,'Species']
plt.scatter(X,Y,color='blue',label='SepalWidthCm')


X=df.loc[:,'PetalLengthCm']
Y=df.loc[:,'Species']
plt.scatter(X,Y,color='red',label='PetalLengthCm')


X=df.loc[:,'PetalWidthCm']
Y=df.loc[:,'Species']
plt.scatter(X,Y,color='black',label='PetalWidthCm')

plt.legend(loc=4, prop={'size':5})
plt.show()



