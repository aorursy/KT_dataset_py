import numpy as np

import pandas as pd
df=pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv") 
df=df[["gender","math score","reading score","writing score"]]
df.head()
df=pd.get_dummies(df,columns=["gender"])
df.head()
df=df.rename(columns={"gender_female":"gender"})
df.drop(["gender_male"],axis=1,inplace=True)
X=df[["math score","writing score"]]
Y=df["gender"]
X.head()
Y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier().fit(X_train,Y_train)
np.sqrt(mean_squared_error(Y_test,knn_model.predict(X_test)))
r2_score(Y_test,knn_model.predict(X_test))
accuracy_score(Y_test,knn_model.predict(X_test))
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
plt.figure(figsize=(16,8))

X_set, y_set = X_test, Y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, knn_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.6, cmap = ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                 c = ListedColormap(('red', 'blue'))(i), label = j)

font={"fontsize":15}

plt.title('KNN Test Seti',fontdict=font)

plt.xlabel('Math + Writing Score',fontdict=font)

plt.ylabel('Gender (Female=1,Male=0)',fontdict=font)

plt.legend(loc=2)

plt.show();