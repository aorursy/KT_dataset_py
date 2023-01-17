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
df_Sugar_cnt = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df_Sugar_cnt
print("Size of the dataset ",df_Sugar_cnt.shape)

print("Checking for the null values ",df_Sugar_cnt.isnull().sum())
import matplotlib.pyplot as plt
x = df_Sugar_cnt['Age']
plt.hist(x, bins=10)
plt.show()
import seaborn as sns
sns.countplot(df_Sugar_cnt['Outcome'],label='count')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(df_Sugar_cnt.drop('Outcome',axis=1))
features = sc.transform(df_Sugar_cnt.drop('Outcome',axis=1))
feature_frame = pd.DataFrame(features,columns=df_Sugar_cnt.columns[:-1])
feature_frame

plt.figure(figsize=(10,10))
sns.heatmap(feature_frame.iloc[:,:].corr(),annot=True)

sns.pairplot(df_Sugar_cnt, hue = 'Outcome')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(feature_frame,df_Sugar_cnt['Outcome'],
                                                    test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report , confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
error_rate = []

for i in range(1,50):
    knn   = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue',linestyle='solid',marker='o',markerfacecolor='green',markersize=10)
plt.title('Error Rate v/s K value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors = 28)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))