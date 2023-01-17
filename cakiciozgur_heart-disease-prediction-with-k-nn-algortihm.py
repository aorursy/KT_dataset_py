import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/heart.csv")
data.head()
data.info()
data.describe()
data.isnull().sum()
data.target.unique()
target_0=data[data.target==0]
target_1=data[data.target==1]
y=data.target.values
x_data=data.drop(["target"],axis=1)
x=(x_data-np.min(x_data))/((np.max(x_data))-(np.min(x_data)))
plt.scatter(target_1.chol,target_1.age,color="red",label="Hasta",alpha=0.5)
plt.scatter(target_0.chol,target_0.age,color="green",label="Hasta Değil",alpha=0.5)
plt.xlabel("Cholestoral")
plt.ylabel("Age")
plt.legend()
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=18)

knn.fit(x_train,y_train)
prediction=knn.predict(x_test)
#prediction
print("{}-NN Score : {}".format(18,knn.score(x_test,y_test)))
score_list=[]
for each in range(5,20):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(5,20),score_list)
plt.show()