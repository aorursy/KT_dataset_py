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
data=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.head()
data.shape
data.isnull().sum()
data.quality.unique()
data.quality.value_counts()
X=data.drop(columns=['quality'])

y=data['quality']
#SMOTE Technique

from imblearn.over_sampling import SMOTE

# transform the dataset

oversample = SMOTE()

X, y = oversample.fit_resample(X, y)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X=scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
from sklearn.neighbors import KNeighborsClassifier



error_rate = []



for i in range(1,20):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_test)

    error_rate.append(np.mean(y_pred != y_test))
import matplotlib.pyplot as plt



plt.figure(figsize=(10,6))

plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
k=2



knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train,y_train)



y_pred=knn.predict(X_test)

from sklearn.metrics import accuracy_score



accuracy_score(y_pred,y_test)
from sklearn.metrics import classification_report,accuracy_score



print(classification_report(y_pred,y_test))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_pred,y_test)