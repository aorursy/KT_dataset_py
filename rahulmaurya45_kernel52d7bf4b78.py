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
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv("../input/titanic/train_and_test2.csv")
df.head()
sns.countplot(x="2urvived",data=df)
sns.countplot(x="Sex",data=df)
sns.countplot(x="2urvived",hue="Sex",data=df)
sns.countplot(x="2urvived",hue="Pclass",data=df)
df.isnull().sum()
sns.boxplot(x="Pclass",y="Age",data=df)
df1=pd.read_csv("../input/titanicdataset-traincsv/train.csv")
df1.head()
df1.drop("Cabin",axis=1,inplace=True)
df1.dropna(inplace=True)
df1.isnull().sum()
sex=pd.get_dummies(df1["Sex"],drop_first=True)
sex.head()
embark=pd.get_dummies(df1["Embarked"],drop_first=True)
embark.head(3)
pc1=pd.get_dummies(df1["Pclass"],drop_first=True)
pc1.head()
df2=pd.concat([df1,sex,embark,pc1],axis=1).head()
df2.drop(["Sex","Embarked","PassengerId","Name","Ticket"],axis=1,inplace=True)
df2

#test and train for Logistic Regression
df2
x=df2.drop(["Survived"],axis=1)
y=df2["Survived"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.6,random_state=1)
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()

log.fit(x_train,y_train)
p=log.predict(x_test)
from sklearn.metrics import classification_report
classification_report(y_test,p)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,p)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,p)
