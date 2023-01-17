# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris=pd.read_csv('../input/iris/Iris.csv')
iris.head()
sns.pairplot(data=iris.drop('Id',axis=1),hue='Species',kind='reg')
corre=iris.drop('Id',axis=1).corr()
sns.heatmap(data=corre)
sns.scatterplot(x="PetalLengthCm",y="SepalLengthCm",data=iris,hue="Species")
sns.jointplot(x="PetalLengthCm",y="SepalLengthCm",data=iris)
from sklearn.model_selection import train_test_split
dat=iris.drop('Id',axis=1)
fea=dat.drop('Species',axis=1)
tar=iris['Species']

x_train,x_test,y_train,y_test=train_test_split(fea,tar,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc=RandomForestClassifier()
rfc.fit(fea,tar)
res=rfc.predict(x_test)
res1=accuracy_score(y_test,res)
print("Random Forest Classifier Success Rate :", "{:.2f}%".format(100*res1))
from sklearn.neighbors import KNeighborsClassifier
scorelist=[]
for i in range(1,100):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    p5=knn.predict(x_test)
    s5=accuracy_score(y_test,p5)
    scorelist.append(round(100*s5, 2))
print("K Nearest Neighbors Top 5 Success Rates:")
print(sorted(scorelist,reverse=True)[:30])