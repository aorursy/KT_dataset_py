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
#read data set 

data=pd.read_csv("../input/melanoma.csv")

data.head()
data=data.drop("Unnamed: 0",axis=1)

data

#delete unnamed area 
data.info()

#data innfo we dot have non null it is good :)

data.describe()

#data statistics
predicseri=["time","sex","age","year","thickness","ulcer"] # our predict serie value

X=data[predicseri]

X
y=data["status"]

y #our target 
from sklearn import linear_model

from sklearn.model_selection import train_test_split 

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=3)

#education with linner reggesion
len(X_train)
modelmultilinear=linear_model.LinearRegression()

modelmultilinear.fit(X_train,y_train)
df2=pd.DataFrame(modelmultilinear.coef_,X.columns,columns=["cofficient"])

df2

#linner reggsion result 
modelmultilinear.intercept_
from sklearn import metrics

import numpy as np

mean_abssulate_Eror=metrics.mean_absolute_error(y_test,modelmultilinear.predict(x_test))

Mean_Squared_Error=metrics.mean_squared_error(y_test, modelmultilinear.predict(x_test))

accurancy=metrics.r2_score(y_test,modelmultilinear.predict(x_test))

df5=pd.DataFrame({"mean_abssulate_Eror":[mean_abssulate_Eror],"Mean_Squared_Error":[Mean_Squared_Error],"accurancy":[accurancy]})

df5 # model control 
accurancy2=metrics.r2_score(y_train,modelmultilinear.predict(X_train))

accurancy2

#all tarin set accurancy
#logisticReggerison

from sklearn.linear_model import LogisticRegression

modellogistic=LogisticRegression()

modellogistic.fit(X_train,y_train)
print("logistic test accurancy:\n",modellogistic.score(x_test,y_test))

print("lineer test accurancy:\n",modelmultilinear.score(x_test,y_test))

#compare 



#test mdoel

print("model test prediciton:\n",modellogistic.predict(x_test.head(10)))

a=np.array(y_test.head(10))

print("real data:\n",a)
from sklearn.metrics import confusion_matrix

y_predict=modellogistic.predict(x_test)

cm=confusion_matrix(y_test,y_predict)
import seaborn as sn

import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

sn.heatmap(cm,annot=True)

plt.xlabel("predict")

plt.ylabel("real")
