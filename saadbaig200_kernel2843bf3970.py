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
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv("/kaggle/input/classification-suv-dataset/Social_Network_Ads.csv")
df.head(10)

sns.countplot(x="Gender",hue = "Purchased",data = df)
sns.countplot(x= "Purchased",data = df)
df['Age'].hist(by=df['Purchased'])
df["EstimatedSalary"].hist(by=df ["Purchased"])
sns.heatmap(df.isnull())
df.drop("User ID",axis = 1,inplace= True)
df.head()
sex = pd.get_dummies(df["Gender"],drop_first=True)
df = pd.concat([df,sex],axis = 1)
df.head()
df.drop("Gender",axis = 1,inplace = True)
df.head()
y=df["Purchased"]
df.drop("Purchased",axis = 1,inplace = True)
X=df
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X= sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
prediction = log.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,prediction )
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)
