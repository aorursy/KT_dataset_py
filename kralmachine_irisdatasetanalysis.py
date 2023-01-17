# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/Iris.csv')
for column in data.columns:
    print(column)
    
print('-'*40)    

data=data.rename(columns={'SepalLengthCm':'sepal_length','SepalWidthCm':'sepal_width','PetalLengthCm':'petal_length','PetalWidthCm':'petal_width','Species':'species'})

for column in data.columns:
    print(column)




print(data.head())
print(data.tail())
print(data.shape)
print(data.info())
print(data.describe())
data['id']=range(1,151)
count1=data[['species','sepal_width']].groupby(['species'],as_index=False).mean().sort_values(by='sepal_width',ascending=False)
print(count1)
print('-'*40)
count2=data[['species','sepal_length']].groupby(['species'],as_index=False).mean().sort_values(by='sepal_length',ascending=False)
print(count2)
print('-'*40)
count3=data[['species','petal_width']].groupby(['species'],as_index=False).mean().sort_values(by='petal_width',ascending=False)
print(count3)
print('-'*40)
count4=data[['species','petal_length']].groupby(['species'],as_index=False).mean().sort_values(by='petal_length',ascending=False)
print(count4)
g=sns.FacetGrid(data,col='species')
g.map(plt.hist,'petal_length',bins=25)
plt.show()
plt.scatter(data['petal_length'],data['petal_width'],marker='*',alpha=.5,color='red',label='petallength')
plt.scatter(data['sepal_length'],data['sepal_width'],marker='o',alpha=.5,color='blue',label='petalwidth')
plt.legend()
plt.show()
data2=data.loc[:,'species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data3=le.fit_transform(data['species'])
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'species'],
                                       c=['green','blue','red'],
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.8,
                                       s = 200,
                                       marker = '.',
                                       edgecolor= "black")
plt.show()
sns.countplot(x='species',data=data)
data.loc[:,'species'].value_counts()
plt.show()
data1=data[data['species']=='Iris-setosa']
x=np.array(data1.loc[:,'sepal_length']).reshape(-1,1)
y=np.array(data1.loc[:,'sepal_width']).reshape(-1,1)
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
data1=data[data['species']=='Iris-versicolor']
x=np.array(data1.loc[:,'sepal_length']).reshape(-1,1)
y=np.array(data1.loc[:,'sepal_width']).reshape(-1,1)
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
data1=data[data['species']=='Iris-virginica']
x=np.array(data1.loc[:,'sepal_length']).reshape(-1,1)
y=np.array(data1.loc[:,'sepal_width']).reshape(-1,1)
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
setosa=data[data.species=='Iris-setosa']
versicolor=data[data.species=='Iris-versicolor']
virginica=data[data.species=='Iris-virginica']
plt.subplot(2,1,1)
plt.plot(setosa.Id,setosa['petal_length'],color='red',label='setosa - PetalLengthCm')

plt.subplot(2,1,2)
plt.plot(versicolor.Id,versicolor['petal_length'],color='green',label='versicolor - PetalLengthCm')

plt.subplot(2,1,2)
plt.plot(virginica.Id,virginica['petal_length'],color='blue',label='virginica - PetalLengthCm')

plt.xlabel('id')
plt.ylabel('petal_length')
plt.legend()
plt.show()
plt.scatter(setosa.iloc[:,2],setosa.iloc[:,3],color='red',label='setosa')
plt.scatter(versicolor.iloc[:,2],versicolor.iloc[:,3],color='green',label='versicolor')
plt.scatter(virginica.iloc[:,2],virginica.iloc[:,3],color='blue',label='virginica')
plt.legend()
plt.xlabel('PetalLength')
plt.ylabel('PetalWidthCm')
plt.title('Scatter Plot')
plt.show()
plt.hist(setosa.iloc[:,2],bins=5)
plt.xlabel('PetalLength values')
plt.ylabel('Frekans')
plt.title('Hist')
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data.iloc[:,0:4],data2,test_size=0.3,random_state=0)
logreg=LogisticRegression(random_state=0)
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

logreg_test=logreg.score(x_test,y_test)
logreg_train=logreg.score(x_train,y_train)

print('Test set score :{:.2f}'.format(logreg_test))
print('Train set score :{:.2f}'.format(logreg_train))
linreg=LinearRegression()
linreg.fit(x_train,y_train)
y_pred=linreg.predict(x_test)

linreg_test=linreg.score(x_test,y_test)
linreg_train=linreg.score(x_train,y_train)

print('Test set score :{:.2f}'.format(linreg_test))
print('Train set score :{:.2f}'.format(linreg_train))
sns.heatmap(cm,annot=True,fmt='d')
plt.show()

svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)


svc_test=svc.score(x_test,y_test)
svc_train=svc.score(x_train,y_train)

print('Test set score :{:.2f}'.format(svc_test))
print('Train set score :{:.2f}'.format(svc_train))
linsvc=LinearSVC()
linsvc.fit(x_train,y_train)
y_pred=linsvc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)

linsvc_test=linsvc.score(x_test,y_test)
linsvc_train=linsvc.score(x_train,y_train)

print('Test set score :{:.2f}'.format(linsvc_test))
print('Train set score :{:.2f}'.format(linsvc_train))
rndf=RandomForestClassifier(random_state=0)
rndf.fit(x_train,y_train)
y_pred=rndf.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)


rndf_test=rndf.score(x_test,y_test)
rndf_train=rndf.score(x_train,y_train)

print('Test set score :{:.2f}'.format(rndf_test))
print('Train set score :{:.2f}'.format(rndf_train))
knc=KNeighborsClassifier()
knc.fit(x_train,y_train)
y_pred=knc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)


knc_test=knc.score(x_test,y_test)
knc_train=knc.score(x_train,y_train)

print('Test set score :{:.2f}'.format(knc_test))
print('Train set score :{:.2f}'.format(knc_train))
neig=np.arange(1,6)
train_accuracy=[]
test_accuracy=[]
for i,k in enumerate(neig):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.append(knn.score(x_test,y_test))

plt.figure(figsize=[5,5])
plt.plot(neig,test_accuracy,label='Testing Accuracy')
plt.plot(neig,train_accuracy,label='Traning Accuracy')
plt.legend()
plt.title('-Value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.show()
print('Best accuracy is {} with K={}'.format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
print(gnb.theta_)
cm=confusion_matrix(y_test,y_pred)
print(cm)


gnb_test=gnb.score(x_test,y_test)
gnb_train=gnb.score(x_train,y_train)

print('Test set score :{:.2f}'.format(gnb_test))
print('Train set score :{:.2f}'.format(gnb_train))
sgdc=SGDClassifier(random_state=0)
sgdc.fit(x_train,y_train)
y_pred=sgdc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)


sgdc_test=sgdc.score(x_test,y_test)
sgdc_train=sgdc.score(x_train,y_train)

print('Test set score :{:.2f}'.format(sgdc_test))
print('Train set score :{:.2f}'.format(sgdc_train)) 

dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)


dtc_test=dtc.score(x_test,y_test)
dtc_train=dtc.score(x_train,y_train)

print('Test set score :{:.2f}'.format(dtc_test))
print('Train set score :{:.2f}'.format(dtc_train))  
sonuclar={'ML':['LogisticRegression','LinearRegression','SVC','LinearSVC','RandomForest','KNN','GaussianNB','SGD','DecisionTree']
,'Test Accuracy':[logreg_test,linreg_test,svc_test,linsvc_test,rndf_test,knc_test,gnb_test,sgdc_test,dtc_test]
,'Train Accuracy':[logreg_train,linreg_train,svc_train,linsvc_train,rndf_train,knc_train,gnb_train,sgdc_train,dtc_train]}

sonuclar1=pd.DataFrame(data=sonuclar,index=range(1,10))
print(sonuclar1)