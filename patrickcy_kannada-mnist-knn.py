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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data=pd.read_csv('../input/Kannada-MNIST/train.csv')
data.head(2)
data.shape
X_data_valid=pd.read_csv('../input/Kannada-MNIST/test.csv')

X_valid = X_data_valid.iloc[:, 1:]
X_valid.shape
X_train= data.values[:600, 1:] # downsample the train data. run too slow in kaggle. 
y_train=data.values[:600, 0]
sns.distplot(y_train)
numbers=['0','1','2','3','4','5','6','7','8','9']
rows=4
for m, number in enumerate(numbers):
    
    indx=[i for i, e in enumerate(y_train) if e==m]
    rand_indx=np.random.choice(indx, rows)
    #print(rand_indx)
    for k in range(rows):
        plt.subplot(rows,10,k*10+m+1)
        plt.figsize=(0.2,0.2)
        plt.imshow(X_train[rand_indx[k],].reshape(28,28))
        plt.axis('off')
        plt.title(y_train[rand_indx[k]])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy=[]
for k in range(20):
    knn=KNeighborsClassifier(n_neighbors=k+1)
    knn.fit(X_train,y_train)
    knn_pred=knn.predict(X_test)
    print(accuracy_score(y_test, knn_pred))
    accuracy.append(accuracy_score(y_test, knn_pred))
    
k=range (1,21)
plt.plot(k, accuracy)
k_best=accuracy.index(max(accuracy))+1
k_best
knn_best=KNeighborsClassifier(n_neighbors=k_best)
knn_best.fit(X_train,y_train)

knn_best_pred_subm=knn_best.predict(X_valid)
sub=pd.DataFrame()
sub['id']=list(X_data_valid.values[0:,0])
sub['label']=knn_best_pred_subm
sub.to_csv("submission.csv", index=False)



