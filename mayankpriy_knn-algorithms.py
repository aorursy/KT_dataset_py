# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# I am using this data set which i took from kaggle named as "Social_Network_ads.csv".

# i will use this data set to pridict who can buy a car if a company launces its new model they can directly contact the employees to advertise there new model

# i am using KNN algorithm of Supervised learning ,which is a classifier type of algorithms

#importing the data set 

data= pd.read_csv("../input/users-of-a-social-networks-who-bought-suv/Social_Network_Ads.csv")

data.head(10)
#now spliting our data  for input and output

#i will take the Age and Estimated Salary columns for the input

input_x=data.iloc[:,[2,3]].values

#now taking the purchased column as output. we can see that purchased columns contain data as binary data(0,1)

#where 0 represent "NO"-it means they donot have car, and 1 represent "YES" -it means they have car

output_x=data.iloc[:,4].values

#now spliting data into Traning Data and Testing Data

train_x,test_x,train_y,test_y=train_test_split(input_x,output_x,test_size=0.30)

#now i am doing feature scaling of data because if we see in our dataset there is a hiuge difference in data of both column,if we compare the both column

#with help of Standard scaler we will bring the data into a range of(-2,2) so we can get get a good prediction

std_scaler=StandardScaler()

train_x=std_scaler.fit_transform(train_x)

test_x=std_scaler.fit_transform(test_x)

#now we will build our model Using KNeighborsClassifier

clssifier_kn=KNeighborsClassifier(n_neighbors=7,metric='minkowski',p=2)

clssifier_kn.fit(train_x,train_y)

#Now pridicting our model

y_pred=clssifier_kn.predict(test_x)

#creating a confusion matrix

c_metrix=confusion_matrix(test_y,y_pred)

c_metrix
# Now i will plot our Model for visualization using matplotlib

# i will plot in two section 1-traing data ,2-testing data

#**********************

#traing data visualization

from matplotlib.colors import ListedColormap

x_set,y_set=train_x,train_y

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),

                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,clssifier_kn.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('aqua','silver')))

plt.xlim=(x1.min(),x1.max())

plt.ylim=(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.title('Logistic Regression (Training Set)')   

plt.xlabel=('Age')

plt.ylabel=('Estimated Salary') 

plt.legend()

plt.show()  
#testing data visualization

from matplotlib.colors import ListedColormap

x_set,y_set=test_x,test_y

x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),

                 np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,clssifier_kn.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('aqua','silver')))

plt.xlim=(x1.min(),x1.max())

plt.ylim=(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):

    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.title('Logistic Regression (Testing Set)')   

plt.xlabel=('Age')

plt.ylabel=('Estimated Salary') 

plt.legend()

plt.show()  