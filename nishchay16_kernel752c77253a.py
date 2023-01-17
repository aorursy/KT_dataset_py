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
data=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
data.head()
data.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data.drop('Outcome',axis=1))
scaled_features = scaler.transform(data.drop('Outcome',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])

df_feat.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,data['Outcome'],

                                                    test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
pred
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
y_test.count()
print(classification_report(y_test,pred))
error_rate = []



# Will take some time

for i in range(1,30):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))

plt.plot(range(1,30),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')