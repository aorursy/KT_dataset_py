

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/column_2C_weka.csv")
data2= pd.read_csv("../input/column_3C_weka.csv")
data.tail(10)
data2.head()
data.info()
f,ax = plt.subplots(figsize =(10,5))
sns.heatmap(data.corr(), annot=True, linewidths = 0.1, fmt= '.1f', ax=ax, cmap = 'Blues')
plt.show()
sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()
DataBinary = data.copy()
DataBinary['class'] = [1 if each == 'Normal' else 0 for each in data['class']]
DataBinary.tail()
x,y = data.loc[:,data.columns != 'class'],data.loc[:, 'class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1 )
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("the prediction is {}".format(prediction))
knn.score(x_test,y_test)
x,y = data.loc[:,data.columns != 'class'],data.loc[:, 'class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1 )

interval = np.arange(1, 25)
test_accuracy = [] 
for i, k in enumerate(interval):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    test_accuracy.append(knn.score(x_test,y_test))
    
#PLOT
plt.figure(figsize=[10,4])
plt.plot(interval, test_accuracy, label = 'Test Accuracy')
plt.legend()
plt.title('Test Accuracy')
plt.xticks(interval)
plt.xlabel("number of neighbors")
plt.ylabel("accuracy")
plt.show()
print('Best accuracy is {} with K = {}'.format(np.max(test_accuracy), 
                                                  1+test_accuracy.index(np.max(test_accuracy))))





