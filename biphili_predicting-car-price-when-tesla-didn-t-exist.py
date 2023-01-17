# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from PIL import Image

%matplotlib inline

import numpy as np

img=np.array(Image.open('../input/tesla-mode3/Tesla.jpg'))

fig=plt.figure(figsize=(10,10))

plt.imshow(img,interpolation='bilinear')

plt.axis('off')

plt.show()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

#plt.style.use('seaborn')

import seaborn as sns

plt.style.use('fivethirtyeight')
car=pd.read_csv('../input/automobile-dataset/Automobile_data.csv')

car.head()
print('Rows     :',car.shape[0])

print('Columns  :',car.shape[1])

print('\nFeatures :\n     :',car.columns.tolist())

print('\nMissing values    :',car.isnull().values.sum())

print('\nUnique values :  \n',car.nunique())
car.shape
#car.isnull().sum()
#car.info
a=car[car['normalized-losses']!='?']

b=(a['normalized-losses'].astype(int)).mean()

car['normalized-losses']=car['normalized-losses'].replace('?',b).astype(int)
a=car[car['body-style']=='sedan']

a['num-of-doors'].value_counts()
a=car['num-of-doors'].map({'two':2,'four':4,'?':4})

car['num-of-doors']=a
a=car[car['price']!='?']

b=(a['price'].astype(int)).mean()

car['price']=car['price'].replace('?',b).astype(int)
a=car[car['horsepower']!='?']

b=(a['horsepower'].astype(int)).mean()

car['horsepower']=car['horsepower'].replace('?',b).astype(int)
a=car[car['bore']!='?']

b=(a['bore'].astype(float)).mean()

car['bore']=car['bore'].replace('?',b).astype(float)
a=car[car['stroke']!='?']

b=(a['stroke'].astype(float)).mean()

car['stroke']=car['stroke'].replace('?',b).astype(float)
a=car[car['peak-rpm']!='?']

b=(a['peak-rpm'].astype(float)).mean()

car['peak-rpm']=car['peak-rpm'].replace('?',b).astype(float)
a=car['num-of-cylinders'].map({'four':4,'five':5,'six':6,'?':4})

car['num-of-doors']=a
car.describe().T
f,ax=plt.subplots(1,2,figsize=(18,8))

car['make'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Make of Car')

#ax[0].set_ylabel('Count')

sns.countplot('make',data=car,ax=ax[1],order=car['make'].value_counts().index)

ax[1].set_title('Make of Car')

#ax[1].set_xticklabels(rotation=30)

plt.show()
car.columns
pd.crosstab(car.make,car['fuel-type'],margins=True).T.style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

car['fuel-type'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Fuel Type')

ax[0].set_ylabel('Count')

sns.countplot('fuel-type',data=car,ax=ax[1],order=car['fuel-type'].value_counts().index)

ax[1].set_title('Fuel Type')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

car['aspiration'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Aspiration Type')

ax[0].set_ylabel('Count')

sns.countplot('aspiration',data=car,ax=ax[1],order=car['aspiration'].value_counts().index)

ax[1].set_title('Aspiration Type')

plt.show()
print('Car makers in the data set are',car['make'].unique())
car[['engine-size','peak-rpm','curb-weight','horsepower','price','highway-mpg']].hist(figsize=(10,8),bins=50,color='b',linewidth='3',edgecolor='k')

plt.tight_layout()

plt.show()
plt.subplots(figsize=(10,6))

ax=car['make'].value_counts().plot.bar(width=0.9,color=sns.color_palette('RdYlGn',20))

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

plt.xticks(rotation='vertical')

plt.xlabel('Car Maker',fontsize=20)

plt.ylabel('Number of Cars',fontsize=20)

plt.title('Cars Count By Manufacturer',fontsize=30)

ax.tick_params(labelsize=15)

#plt.yticks(rotation='vertical')

plt.show()

plt.show()
fig = plt.figure(figsize=(15, 10))

ax=sns.countplot(car['make'],palette='dark',edgecolor='k',linewidth=2,order = car['make'].value_counts().index)

plt.xticks(rotation='vertical')

plt.xlabel('Car Maker',fontsize=20)

plt.ylabel('Number of Cars',fontsize=20)

plt.title('Cars Count By Manufacturer',fontsize=30)

ax.tick_params(labelsize=15)

#plt.yticks(rotation='vertical')

plt.show()
print('Different types of cars',car['body-style'].unique())
fig = plt.figure(figsize=(15, 10))

cars_type=car.groupby(['body-style']).count()['make']

ax=cars_type.sort_values(ascending=False).plot.bar(edgecolor='k',linewidth=2)

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1),fontsize=11)

plt.xticks(rotation='vertical')

plt.xlabel('Body Type',fontsize=20)

plt.ylabel('Number of Cars',fontsize=20)

plt.title('Count of Cars by Body Type',fontsize=30)

ax.tick_params(labelsize=15)

#plt.yticks(rotation='vertical')

plt.show()
from matplotlib.pyplot import plot

fig = plt.figure(figsize=(25, 25))

a=car.groupby(['body-style','make']).count().reset_index();

a=a[['make','body-style','symboling']]

a.columns=['make','style','count']

a=a.pivot('make','style','count')

a.dropna(thresh=3).plot.bar(width=0.85);

#plot.bar()

plt.ioff()

plt.show()
plt.figure(1)

plt.subplot(221)

ax1=car['engine-type'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='orange',edgecolor='k',linewidth=2)

plt.title("Number of Engine Type frequency diagram")

plt.ylabel('Number of Engine Type',fontsize=15)

ax1.tick_params(labelsize=15)

plt.xlabel('engine-type',fontsize=15);





plt.subplot(222)

ax2=car['num-of-doors'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='purple',edgecolor='k',linewidth=2)

plt.title("Number of Door frequetncy diagram")

plt.ylabel('Number of Doors',fontsize=15)

ax2.tick_params(labelsize=15)

plt.xlabel('num-of-doors',fontsize=15);



plt.subplot(223)

ax3=car['fuel-type'].value_counts(normalize= True).plot(figsize=(10,8),kind='bar',color='green',edgecolor='k',linewidth=2)

plt.title("Number of Fuel Type frequency diagram")

plt.ylabel('Number of vehicles',fontsize=15)

plt.xlabel('fuel-type',fontsize=15)

ax2.tick_params(labelsize=15)



plt.subplot(224)

ax4=car['body-style'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='red',edgecolor='k',linewidth=2)

plt.title("Number of Body Style frequency diagram")

plt.ylabel('Number of vehicles',fontsize=15)

plt.xlabel('body-style',fontsize=15);

plt.tight_layout()

plt.show()

fig = plt.figure(figsize=(15, 10))

mileage=car.groupby(['make']).mean()

mileage['avg-mpg']=((mileage['city-mpg']+mileage['highway-mpg'])/2)

ax=mileage['avg-mpg'].sort_values(ascending=False).plot.bar(edgecolor='k',linewidth=2)

plt.xticks(rotation='vertical')

plt.xlabel('Car Maker',fontsize=20)

plt.ylabel('Number of Cars',fontsize=20)

plt.title('Fuel Economy of Car Makers',fontsize=30)

ax.tick_params(labelsize=20)

#plt.yticks(rotation='vertical')

plt.show()

plt.show()
plt.rcParams['figure.figsize']=(23,10)

ax=sns.factorplot(data=car, x="num-of-cylinders", y="horsepower");

#ax.set_xlabel('Number of Cyliner',fontsize=30)

#ax.set_ylabel('Horse Power',fontsize=30)

#plt.title('Horse Power Vs Num of Cylinder',fontsize=40)

#ax.tick_params(axis='x',labelsize=20,rotation=90)

plt.ioff()
plt.rcParams['figure.figsize']=(23,10)

ax = sns.boxplot(x="make", y="price", data=car,width=0.8,linewidth=5)

ax.set_xlabel('Make of Car',fontsize=30)

ax.set_ylabel('Price in $',fontsize=30)

plt.title('Price of Car Based on Make',fontsize=40)

ax.tick_params(axis='x',labelsize=20,rotation=90)
sns.factorplot(data=car, y="price", x="body-style" , hue="fuel-type" ,kind="point")

plt.xlabel('Type of Engine',fontsize=20)

plt.ylabel('Price in $',fontsize=20)

plt.title('Price Vs Engine Type',fontsize=20)

plt.tick_params(axis='x',labelsize=10,rotation=90)
plt.rcParams['figure.figsize']=(23,10)

ax=sns.boxplot(x='drive-wheels',y='price',data=car,width=0.8,linewidth=5)

ax.set_xlabel('Make of Car',fontsize=30)

ax.set_ylabel('Price in $',fontsize=30)

plt.title('Price of Car Based on Make',fontsize=40)

ax.tick_params(axis='x',labelsize=20,rotation=90)
import seaborn as sns

plt.figure(figsize=(20,10))

sns.heatmap(car.corr(),annot=True,cmap='summer');
ax = sns.pairplot(car[["width", "curb-weight","engine-size","horsepower","highway-mpg","fuel-type","price",]], hue="fuel-type",palette='dark') #diag_kind="hist"
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split # for spliting the data into training and test set

from sklearn import metrics # for validating the accuracy of the model
train,test=train_test_split(car,test_size=0.05)

train.head()
X_train=train[['curb-weight','engine-size','horsepower','width']]

#X_train = train.drop('price',axis=1)

y_train=train.price
X_test=test[['curb-weight','engine-size','horsepower','width']]

y_test=test.price
model=LinearRegression()

model.fit(X_train,y_train)

prediction=model.predict(X_test)
y_test
Line=prediction.astype(int)

Line
model=LogisticRegression()

lm=model.fit(X_train,y_train)

prediction=model.predict(X_test)

#print('Accuracy of the Logistic Regression is:',metrics.accuracy_score(prediction,y_test))

logi=prediction.astype(int)

logi
model=DecisionTreeClassifier()

model.fit(X_train,y_train)

prediction=model.predict(X_test)
DTree=prediction.astype(int)

DTree