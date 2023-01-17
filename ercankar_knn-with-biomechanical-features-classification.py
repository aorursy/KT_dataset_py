import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plot your data





import os

print(os.listdir("../input"))

os.chdir("../input")

%matplotlib inline
df= pd.read_csv('column_2C_weka.csv')
df.head()
df.shape
df['class'].unique()
df['class'].value_counts()
df.hist()
df.columns
X=df.iloc[:,0:6].values

y=df['class'].values

X
from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))

#X=(X-np.min(X))/(np.max(X)/np.min(X))

X
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
neigh= KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train) 

# our distance metric is Euclidean metric. more: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
yhat = neigh.predict(X_test)

yhat
from sklearn import metrics

print("Train Set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))

print("Test Set Accuracy: ",metrics.accuracy_score(y_test,yhat))
accuracy_list= []

for each in range(1,25):

    neigh_fit= KNeighborsClassifier(n_neighbors=each).fit(X_train,y_train)

    yhat=neigh_fit.predict(X_test)

    accuracy_list.append(metrics.accuracy_score(y_test,yhat))



plt.title('k-NN Varying number of neighbors')

plt.plot(range(1,25),accuracy_list)

plt.legend(loc='best',title="ACCURACY")

plt.xlabel('k values')

plt.ylabel('accuracy')

plt.show()