# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris = pd.read_csv("../input/iris/Iris.csv") #load the dataset

iris.head()
iris.drop('Id',axis=1,inplace=True)
import matplotlib.pyplot as pt

import seaborn as sns

%matplotlib inline
sns.scatterplot(x='PetalLengthCm',y='PetalWidthCm',hue='Species',data=iris)
sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=iris)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler_fit=scaler.fit(iris.drop('Species',axis=1))
new_data=scaler.transform(iris.drop('Species',axis=1))
new_dataframe=pd.DataFrame(new_data,columns=iris.columns[:-1])
new_dataframe.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X=new_dataframe

y=iris['Species']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
X_train.head()
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
from sklearn.metrics import classification_report ,accuracy_score
classification_report(predictions,y_test)
accuracy_score(predictions,y_test)
a=[]

for i in range(1,40):

    model=KNeighborsClassifier(n_neighbors=i)

    model.fit(X_train,y_train)

    predictions=model.predict(X_test)

    score=accuracy_score(y_test,predictions)

    a.append(score)

    
m=range(1,40)

plt.plot(m,a)