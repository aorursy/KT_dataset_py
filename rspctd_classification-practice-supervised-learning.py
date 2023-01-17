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
data=pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

data.head()
data['class'].value_counts()
abnormal=data[data['class']=='Abnormal']
normal=data[data['class']=='Normal']

sns.heatmap(data.corr(),annot=True)
plt.show()
data.columns
plt.scatter(abnormal['pelvic_incidence'],abnormal['pelvic_radius'],c='r',label='abnormal',alpha=0.5)
plt.scatter(normal['pelvic_incidence'],normal['pelvic_radius'],c='g',label='normal',alpha=0.6)
plt.xlabel('pelvic incidence')
plt.ylabel('pelvic_radius')
plt.show()
data['class']=[0 if each=='Abnormal'else 1 for each in data['class']]
y=data['class'].values
x_data=data.drop(['class'],axis=1)
x=(x_data-np.min(x_data)) / (np.max(x_data)-np.min(x_data)) # normalization
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

print('Accuracy of Logistic Regression Classification: ',lr.score(x_test,y_test))
data.head()
#train-test-split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
#knn model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=26) #k_value
knn.fit(x_train,y_train)
predict=knn.predict(x_test)
print('{}nn score:  {}'.format(2,knn.score(x_test,y_test)))
score=[]

for each in range(1,50):
    knn2=knn=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    knn2.score(x_test,y_test)
    
    score.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,50),score,c='b')
plt.xlabel('k_value')
plt.ylabel('Accuracy')
plt.show()
    
    
from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.svm import SVC
svm=SVC(random_state=1)
svm.fit(x_train1,y_train1)

print('SVM prediction score :',svm.score(x_test1,y_test1))
from sklearn.model_selection import train_test_split
x_train2,x_test2,y_train2,y_test2=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train2,y_train2)
print('Accuracy of Naive Bayes Algorithm: ',nb.score(x_test2,y_test2))
from sklearn.model_selection import train_test_split
x_train3,x_test3,y_train3,y_test3=train_test_split(x,y,test_size=0.2,random_state=1)

y_test3
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(random_state=1)
dtc.fit(x_train3,y_train3)
print('Accuracy of Decision Tree Regression: ', dtc.score(x_test3,y_test3))
from sklearn.model_selection import train_test_split
x_train4,x_test4,y_train4,y_test4=train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100,random_state=1) # n_estimators bizim modelimizde kac tane tree olmasini istiyorsak ona gore secmeliyiz. Sectigimiz bu tree lerinde her defasinda aynilarinin olmasi icin random_state=1 yapiyoruz.
rfc.fit(x_train4,y_train4)
print('Accuracy of Random Forest Classification : ',rfc.score(x_test4,y_test4))
from sklearn.model_selection import train_test_split
x_train5,x_test5,y_train5,y_test5=train_test_split(x,y,test_size=0.2,random_state=1)
y_head=rfc.predict(x_test5)  # fit edilen modelimize, x_test verilerini koyarak modelimizin tahminlerini buluyoruz(random forest icin bunun accuracy sini yukarida bulduk: 0.838)
y_true=y_test5
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true,y_head)
cm
f,ax= plt.subplots(figsize=(7,7))
sns.heatmap(cm,annot=True,fmt='.0f')
plt.xlabel('y_predicted')
plt.ylabel('y_true')
plt.show()
