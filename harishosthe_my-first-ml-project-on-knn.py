%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
fruits=pd.read_table("../input/fruit_data_with_colors.txt")
fruits.tail(10)
fruits.shape
fruits.info()
f1=dict(zip(fruits["fruit_label"].unique(),fruits["fruit_name"].unique()))
f1
fruits.columns
X=fruits[['mass', 'width', 'height']]
y=fruits["fruit_label"]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
sns.pairplot(X_train)
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(X_train["width"],X_train["height"],X_train["mass"],c=y_train,marker="o",s=100)
ax.set_xlabel("Width")
ax.set_ylabel("Height")
ax.set_zlabel("mass")
y_train.unique()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
predict=knn.predict([[2,4.2,3]])
f1[predict[0]]
k_range=range(1,20)
scores=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    scores.append(knn.score(X_test,y_test))
    
plt.figure()
plt.xlabel("k")
plt.ylabel("accuracy")
plt.scatter(k_range,scores)