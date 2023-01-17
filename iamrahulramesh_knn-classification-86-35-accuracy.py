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
data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

import matplotlib.pyplot as plt

import seaborn as sns

print(plt.style.available)
plt.style.use('ggplot')
data.head()
data.info()
data.describe()
color_list = ['red' if i == 'Abnormal' else 'green' for i in data.loc[:,'class']]



pd.plotting.scatter_matrix (data.loc[:,data.columns != 'class'],

                           c= color_list,

                           figsize=[15,15],

                           diagonal ='hist',

                           alpha =0.5,

                           s =100,

                           marker ='o',

                           edgecolor ='black')



plt.show()
data.loc[:,'class'].value_counts()
sns.countplot(x='class',data=data)
from sklearn.neighbors import KNeighborsClassifier



knn =KNeighborsClassifier(n_neighbors=3)



x= data.loc[:,data.columns !='class']

y= data.loc[:,'class']



knn.fit(x,y)



prediction = knn.predict(x)



prediction
from sklearn.model_selection import train_test_split



x= data.loc[:,data.columns !='class']

y= data.loc[:,'class']



knn =KNeighborsClassifier(n_neighbors=3)



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.35,random_state = 7)



knn.fit(x_train,y_train)



prediction = knn.predict(x_test)



print('Accuracy of KNN with (K=3):',knn.score(x_test,y_test))

kvalue =np.arange(1,25)



train_accuracy =[]

test_accuracy =[]



for i,k in enumerate(kvalue):

    

    knn =KNeighborsClassifier(n_neighbors=k)

    

    knn.fit(x_train,y_train)

    

    train_accuracy.append(knn.score(x_train,y_train))

    

    test_accuracy.append(knn.score(x_test,y_test))

    

print(train_accuracy)

print(test_accuracy)

    
plt.figure(figsize =[15,7])



plt.plot(kvalue, train_accuracy,label = 'Training Accuracy')

plt.plot(kvalue, test_accuracy,label ='Testing Accuracy')



plt.legend(loc='best')

plt.title('Kvalue vs Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')



plt.xticks(kvalue)

plt.show()
print('Best accuracy is {} with K= {}'.format(np.max(test_accuracy),1+ test_accuracy.index(np.max(test_accuracy))))