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
# import Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import Dataset:
df=pd.read_csv("../input/Iris.csv")
df.head()
df.info()
# drop the id column:
df=df.drop('Id',axis=1)
df.head()
#visualization:
sns.pairplot(df,hue='Species')
sns.distplot(df['SepalLengthCm'],bins=50,kde=False)
sns.distplot(df['SepalWidthCm'],bins=50,kde=False)
sns.distplot(df['PetalLengthCm'],bins=50,kde=False)
sns.distplot(df['PetalWidthCm'],bins=50,kde=False)
df.boxplot(column='SepalLengthCm',by='Species')
df.boxplot(column='SepalWidthCm',by='Species')
df.boxplot(column='PetalLengthCm',by='Species')
df.boxplot(column='PetalWidthCm',by='Species')

sns.catplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=df)
sns.catplot(x='PetalLengthCm',y='PetalWidthCm',hue='Species',data=df)
# spliting the dataset into dependent and independent:
X=df.iloc[:,:-1].values
y=df.iloc[:,4].values
y=y.reshape(len(y),1)
X
y

# spliting the dataset into train and test:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# fitting the dataset into the SVM:
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,y_train)
# predication of new result:
y_pred=classifier.predict(X_test)
y_pred
y_test
# make the matric:
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
cm
cr=classification_report(y_test,y_pred)
cr
(13+15+9)/38
