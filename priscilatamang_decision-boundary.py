# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as ml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()
data=df[['SepalLengthCm','PetalLengthCm','Species']]
data
#data['Species'].replace({'Iris-setosa','0', 'Iris-versicolor','1', 'Iris-virginica','2', inplace=True
data['Species'].replace('Iris-setosa','0',inplace=True)
data['Species'].replace('Iris-versicolor','1',inplace=True)
data['Species'].replace('Iris-virginica','2',inplace=True)
data
#Extracting X and Y
X=data.iloc[:,0:2].values
Y=data.iloc[:,2].values

#Train Test Split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=9)

scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

print("The shape of X_train:",X_train.shape)
print("The shape of Y_train:",Y_train.shape)
#Training the model by applying Decision Tree Classifier Algorithm and passing X_train and Y_train to it.
classifier=DecisionTreeClassifier()
classifier.fit(X_train, Y_train)
#Training the model by applying K-Nearest Neighbors Algorithm and passing X_train and Y_train to it.
knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(X_train,Y_train)
A = np.arange(start = X_train[:,0].min()-1,stop=X_train[:,0].max()+1,step=0.01)
B=np.arange(start=X_train[:,1].min()-1,stop=X_train[:,1].max()+1,step=0.01)

print("The shape of A:", A.shape)
print("The shape of B:", B.shape)

XX,YY = np.meshgrid(A,B)

print("The shape of XX:", XX.shape)
print("The shape of YY:", YY.shape)

#Total points to be plotted in CO-ordinate system
total=528*644
print("The total points:", total)

print(XX)
print(YY)
#Predicting for First row.
print(XX[0][0], YY[0][0])
knn.predict(np.array([-2.961714943193718 , -2.640687719884138]).reshape(1,2))
#Using Ravel Function to transform Higher dimensional arrays to 1 D
np.array([XX.ravel(),YY.ravel()]).shape

#Transposing it to get a separate column of SepalLength and PetalLength in a list and storing it in a variable
input_array=np.array([XX.ravel(),YY.ravel()]).T
print("The shape of input_array:",input_array.shape)
#Predicting and storing it in a variable
labels=knn.predict(input_array)

print("The shape of labels:",labels.shape)
labels
labels.shape
#Since Labels is 1D and XX & YY are 2D. So, lables has to be reshaped in the shape of either XX or YY
plt.contourf(XX,YY,labels.reshape(XX.shape))
plt.xlabel("Sepal Length in Cm")
plt.ylabel("Petal Length in Cm")
plt.title("VISUALIZING DECISION BOUNDARY FOR IRIS DATASET")
plt.show()
plt.contourf(XX,YY,labels.reshape(XX.shape),alpha=0.75)
plt.scatter(X_train[:,0],X_train[:,1])