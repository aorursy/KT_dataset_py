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

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/iris/Iris.csv')
df.head()
df.info()
df.head()
df['Species'].unique()
sns.scatterplot('SepalLengthCm','SepalWidthCm',hue='Species',data=df)
sns.scatterplot('SepalLengthCm','PetalLengthCm',hue='Species',data=df)
sns.scatterplot('SepalWidthCm','PetalLengthCm',hue='Species',data=df)
df['Species'] = df['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
df.isnull().sum()
df.columns
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

y = df['Species']
from sklearn.preprocessing import StandardScaler

std = StandardScaler()

X_scales = std.fit_transform(X)
X.corr()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state = 42 )
from sklearn.cluster import KMeans

error = []

for k in range(1,40):

    model = KMeans(n_clusters=k)

    model.fit(X_train)

    error.append(model.inertia_)

    

plt.plot(range(1,40),error,)

plt.xlabel('Number of K values')

plt.ylabel('wcss')

plt.show()

    
#Hence using k=3 as final.

model = KMeans(n_clusters=3)

model.fit(X_train)

result = model.predict(X_test)



    

prediction = pd.DataFrame({'Prediction':result,'True Values':y_test})
prediction
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(result,y_test))
print(classification_report(result,y_test))