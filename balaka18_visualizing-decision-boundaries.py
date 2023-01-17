# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as ml

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, roc_auc_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')

iris.head()
iris.drop(columns=['Id'],inplace=True)

iris = iris.rename(columns = {'SepalLengthCm':'SL','SepalWidthCm':'SW','PetalLengthCm':'PL','PetalWidthCm':'PW'})

iris['Species'].replace('Iris-setosa','0',inplace=True)

iris['Species'].replace('Iris-versicolor','1',inplace=True)

iris['Species'].replace('Iris-virginica','2',inplace=True)

iris
data = iris[['SL','PL','Species']]

data
X = data.iloc[:,:2].values

Y = data.iloc[:,-1].values

scaler = StandardScaler()



x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=9)



x_train=scaler.fit_transform(x_train)

x_test=scaler.fit_transform(x_test)
# 1

dt = DecisionTreeClassifier(criterion='entropy',splitter='best')

dt.fit(x_train,y_train)
# 2

knn = KNeighborsClassifier(n_neighbors = 13, algorithm = 'auto')

knn.fit(x_train,y_train)
xc = np.arange(start = x_train[:,0].min()-1, stop = x_train[:,0].max()+1, step = 0.01)

yc = np.arange(start = x_train[:,1].min()-1, stop = x_train[:,1].max()+1, step = 0.01)

print(xc.shape,yc.shape)
XX,YY = np.meshgrid(xc,yc)

print("XX : ", XX.shape,", YY : ",YY.shape)
528 * 644
data_test = np.array([XX.ravel(),YY.ravel()]).T

data_test.shape
# 1

y_pred1 = dt.predict(data_test)

print(y_pred1.shape)

y_pred1
# 2

y_pred2 = knn.predict(data_test)

print(y_pred2.shape)

y_pred2
# 1

plt.figure(figsize=(20,10),dpi=85)

Z1 = y_pred1.reshape(XX.shape)

plt.contourf(XX,YY,Z1)

plt.show()
# 2

plt.figure(figsize=(20,10),dpi=85)

Z2 = y_pred2.reshape(XX.shape)

plt.contourf(XX,YY,Z2)

plt.show()
# Assign colors to labels trained on

colors_dict = {'0':'white','1':'pink','2':'red'}

y_train = [colors_dict[i] for i in y_train]

y_train
# Plotting

# 1

plt.figure(figsize=(20,10),dpi=85)

plt.contourf(XX,YY,y_pred1.reshape(XX.shape),alpha=0.65)

plt.scatter(x=x_train[:,0],y=x_train[:,1],c=y_train,marker='o')

plt.xlabel("Sepal Length (Cm)")

plt.ylabel("Petal Length (Cm)")

plt.title('VISUALIZING DECISION BOUNDARY FOR DECISION TREE CLASSIFIER(CRITERION = ENTROPY)')

plt.show()
# 2

plt.figure(figsize=(20,10),dpi=85)

plt.contourf(XX,YY,y_pred2.reshape(XX.shape),alpha=0.65)

plt.scatter(x=x_train[:,0],y=x_train[:,1],c=y_train,marker='o')

plt.xlabel("Sepal Length (Cm)")

plt.ylabel("Petal Length (Cm)")

plt.title('VISUALIZING DECISION BOUNDARY FOR KNN CLASSIFIER')

plt.show()