# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#visualization



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
data=pd.read_csv(r"../input/heart-disease-prediction/datasets_33180_43520_heart.csv")
data
#check the shape of the dataset

data.shape
#check the datatypes of the dataset

data.dtypes
data.info()
#check the null value

data.isnull().sum()
data.head()
data.tail()
corr=data.corr()

plt.figure(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,cmap='coolwarm')

plt.show()
sns.countplot(data.target)

plt.show()
sns.pairplot(data)

plt.show()
sns.distplot(data.age[data.target==0])

sns.distplot(data.age[data.target==1])

plt.legend(['0','1'])

plt.show()
sns.countplot(data.sex,hue=data.target)

plt.show()
sns.countplot(data.cp,hue=data.target)

plt.show()
sns.distplot(data.trestbps[data.target==0])

sns.distplot(data.trestbps[data.target==1])

plt.legend(['0','1'])

plt.show()
sns.distplot(data.chol[data.target==0])

sns.distplot(data.chol[data.target==1])

plt.legend(['0','1'])

plt.show()
sns.distplot(data.thalach[data.target==0])

sns.distplot(data.thalach[data.target==1])

plt.legend(['0','1'])

plt.show()
sns.distplot(data.oldpeak[data.target==0])

sns.distplot(data.oldpeak[data.target==1])

plt.legend(['0','1'])

plt.show()
sns.countplot(data.fbs,hue=data.target)

plt.show()
sns.countplot(data.restecg,hue=data.target)

plt.show()
sns.countplot(data.exang,hue=data.target)

plt.show()
sns.countplot(data.slope,hue=data.target)

plt.show()
sns.countplot(data.ca,hue=data.target)

plt.show()
sns.countplot(data.thal,hue=data.target)

plt.show()
sns.catplot(x="cp", y="chol",hue="sex",data=data, kind="bar")

plt.show()
# Show the results of a linear regression within each dataset

sns.lmplot(x="trestbps", y="chol",data=data,hue="cp")

plt.show()
#input and output selection

ip=data.drop(['target'],axis=1)

op=data['target']
from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.3)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(xtr)

xtr=sc.transform(xtr)

xts=sc.transform(xts)
from sklearn.linear_model import LogisticRegression

alg=LogisticRegression()
#train the algorithm with the training data

alg.fit(xtr,ytr)

yp=alg.predict(xts)
from sklearn import metrics

cm=metrics.confusion_matrix(yts,yp)

print(cm)
accuracy=metrics.accuracy_score(yts,yp)

print(accuracy)
precission=metrics.precision_score(yts,yp)

print(precission)
recall=metrics.recall_score(yts,yp)

print(recall)
from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.2)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(xtr)

xtr=sc.transform(xtr)

xts=sc.transform(xts)
from sklearn.naive_bayes import GaussianNB

GNB=GaussianNB()

GNB.fit(xtr,ytr)

yp=GNB.predict(xts)
from sklearn import metrics

cm=metrics.confusion_matrix(yts,yp)

print(cm)
accuracy=metrics.accuracy_score(yts,yp)

print(accuracy)
recall=metrics.recall_score(yts,yp)

print(recall)
#KNN algorithm the nearest distance is calculated

from sklearn.neighbors import KNeighborsClassifier



neighbors=np.arange(1,9)

train_accuracy=np.empty(len(neighbors))

test_accuracy=np.empty(len(neighbors))



for i,k in enumerate(neighbors):

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(xtr,ytr)

    train_accuracy[i]=knn.score(xtr,ytr)

    test_accuracy[i]=knn.score(xts,yts)



plt.xlabel('neighbors of number')

plt.ylabel('accuracy')

plt.title('k-NN Varying number of neighbors')

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')

plt.plot(neighbors, train_accuracy, label='Training accuracy')

plt.legend()

plt.show()

from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.3)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(xtr)

xtr=sc.transform(xtr)

xts=sc.transform(xts)
knn=KNeighborsClassifier(n_neighbors=4)

knn.fit(xtr,ytr)

yp=knn.predict(xts)
from sklearn import metrics

cm=metrics.confusion_matrix(yts,yp)

print(cm)
accuracy=metrics.accuracy_score(yts,yp)

print(accuracy)
recall = metrics.recall_score(yts,yp,average='macro')

print(recall)
from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.3)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(xtr)

xtr=sc.transform(xtr)

xts=sc.transform(xts)
from sklearn import svm



alg=svm.SVC(C=30,gamma=0.03)



#train the algorithm with training data

alg.fit(xtr,ytr)

yp=alg.predict(xts)
from sklearn import metrics

cm=metrics.confusion_matrix(yts,yp)

print(cm)
from sklearn import metrics

accuracy=metrics.accuracy_score(yts,yp)

print(accuracy)
recall = metrics.recall_score(yts,yp)

print(recall)