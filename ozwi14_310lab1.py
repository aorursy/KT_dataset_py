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
data = pd.read_csv('../input/wisconsin_breast_cancer.csv')

cdata = data.fillna(0)

cdata.head()
cdata.info()
import seaborn as sns

sns.pairplot(data=cdata, hue ='class' ,palette='Set2')
from sklearn.model_selection import train_test_split

x = cdata.iloc[:, 1:-1]

y = cdata.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

print(x_train.shape, y_train.shape)



from sklearn.svm import SVC

model=SVC()

model.fit(x_train, y_train)



pred = model.predict(x_test)

print(pred[:10])

print(y_test[:10])



from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))
from sklearn.model_selection import train_test_split

x_sub = cdata.loc[:,['size','thickness','chromatin','mitosis']]

y = cdata.iloc[:,-1]

xsub_train, xsub_test, y_train, y_test = train_test_split(x_sub,y,test_size=0.2)

print(x_train.shape, y_train.shape)



from sklearn.svm import SVC

model=SVC()

model.fit(xsub_train, y_train)



pred = model.predict(xsub_test)

print(pred[:10])

print(y_test[:10])



from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))
from sklearn.neighbors import KNeighborsClassifier  

classifier = KNeighborsClassifier(n_neighbors=5)  

classifier.fit(x_train, y_train)  

 

knnpred =  classifier.predict(x_test) 

print(knnpred[:10])

print(y_test[:10])



from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, knnpred))

print(classification_report(y_test, knnpred))