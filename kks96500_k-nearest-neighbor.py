#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

#importing dataset



dataset=pd.read_csv('../input/Social_Network_Ads.csv')

X=dataset.iloc[:,[2,3]].values

y=dataset.iloc[:,4].values
X
X.shape
y
y.shape
#splitting the dataset into training set and test set (300:100)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
#Applying feature scaling to the independent variables  

from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.fit_transform(X_test)
#fitting the K nearest neighbour model to the training set

from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

classifier.fit(X_train,y_train)

#predicting the new results 

y_pred=classifier.predict(X_test)
y_pred
y_pred.shape
#making the confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
cm
 #visualising the training set results 

from matplotlib.colors import ListedColormap

X_set,y_set=X_train,y_train

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start

                            =X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75

            ,cmap=ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())

plt.ylim(X2.min(),X2.max())

for i ,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.title('K-NN (training set)')

plt.xlabel('Age')

plt.ylabel('Expected Salary')

plt.legend()

plt.show()

#visualising the test set results 

from matplotlib.colors import ListedColormap

X_set,y_set=X_test,y_test

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start

                            =X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75

            ,cmap=ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())

plt.ylim(X2.min(),X2.max())

for i ,j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.title('K-NN(test set)')

plt.xlabel('Age')

plt.ylabel('Expected Salary')

plt.legend()

plt.show()
