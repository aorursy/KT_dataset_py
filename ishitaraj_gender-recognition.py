# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/voicegender/voice.csv')
df.head(2)
df.info()
df.describe()
df['label']=[1 if each=='female' else 0 for each in df['label']]
df.head(3)
df.tail(3)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df.drop('label',axis=1))
scaled_features=scaler.transform(df.drop('label',axis=1))
scaled_features
df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head(3)
from sklearn.model_selection import train_test_split
X=df_feat
y=df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12,7))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='green',markersize=10)
plt.title('error rate vs k')
plt.xlabel('K')
plt.ylabel('Error rate')
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
