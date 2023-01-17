# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
data
data.shape

data.columns
data.head()
data.tail()
data.sample(5)
data.head(10)
x=data.describe()
x=x.iloc[1:,:]
x.style.background_gradient(cmap="Wistia")
data.isnull().sum().sum()
data.dtypes
data.info()
import matplotlib.pyplot as plt
import seaborn as sb
sb.barplot(data.species,data.sepal_length)
sb.lineplot(data.species,data.sepal_length)
sb.scatterplot(data.sepal_width,data.sepal_length)
data.iloc[:,1:].plot(kind="line")
!pip install dabl
import dabl as db
db.plot(data,target_col='species')
email
adhar
pan
mobile
health
bmi
age
salary
data.profile_report()
plt.rcParams["figure.figsize"]=(18,6)
plt.style.use("classic")
sb.stripplot(data.species,data.petal_length)
plt.title("species and petal length comp",fontsize=20) 
plt.grid()
plt.show()
plt.style.available
sb.distplot(data.sepal_length)
sb.countplot(data.species)
data["species"].value_counts().plot(kind='pie',colors=['pink','black','blue'])
plt.axis("off")
data.info()
data.head()
features=data.iloc[:,:4]
print(features)
target=data.iloc[:,4]
print(target)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=0)
print(features.shape)
print(target.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
x_test.shape
from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.linear_model import LogisticRegression 
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))

from sklearn.metrics import confusion_matrix,classification_report
cr=classification_report(y_test,y_pred)
print(cr)
cm=confusion_matrix(y_test,y_pred)
sb.heatmap(cm,annot=True)
print(y_pred)
