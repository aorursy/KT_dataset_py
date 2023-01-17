# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset=pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
dataset.head()
dataset.info()
print(plt.style.available)
plt.style.use('classic')
dataset.describe()
color_bar=['red' if i=='Abnormal' else 'green' for i in dataset.loc[:,'class']]
pd.plotting.scatter_matrix(dataset.loc[:,dataset.columns!='class'],
                          c=color_bar,
                          diagonal='hist',
                          figsize=[15,15],
                          edgecolor='black',
                          marker='*',
                          s=200,
                          alpha=0.5)
sns.countplot(x='class',data=dataset)
dataset.loc[:,'class'].value_counts()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
x,y=dataset.loc[:,dataset.columns!='class'],dataset.loc[:,'class']
knn.fit(x,y)
prediction=knn.predict(x)
print('Prediction:{}'.format(prediction))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=1)
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)
print('accuracy is:',knn.score(x_test,y_test))
#print('prediction is:{}'.format(prediction))
#model complexity
neigh=np.arange(1,25)
train_list=[]
test_list=[]

for i,k in enumerate(neigh):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    train_list.append(knn.score(x_train,y_train))
    test_list.append(knn.score(x_test,y_test))
    
plt.figure(figsize=[10,10])
plt.plot(neigh,train_list,label='train accuracy')
plt.plot(neigh,test_list,label='test accuracy')
plt.legend()
plt.title('-value vs accuracy')
plt.xlabel('number of neighbors')
plt.ylabel('accuracy')
plt.savefig('graphknn.png')
plt.show()

print('the test accuracy is {} and k is {}'.format(np.max(test_list), 1+test_list.index(np.max(test_list))))
data1=dataset[dataset['class']=='Abnormal']

x=np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)
y=np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)

plt.figure(figsize=[10,10])
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
predict_space= np.linspace(min(x),max(x)).reshape(-1,1)
lin_reg.fit(x,y)
predicted= lin_reg.predict(predict_space)
print('r^2 :', lin_reg.score(x,y))
plt.plot(predict_space,predicted, linewidth=3, color='black')
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
from sklearn.model_selection import cross_val_score
reg=LinearRegression()
k=5
cv_result=cross_val_score(reg,x,y,cv=k)
print('the cross validation score is: ',cv_result)
print('average score is: ', np.sum(cv_result)/k)
