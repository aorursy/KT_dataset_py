# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/Iris.csv')

data.head()
data.shape
data.info()
data["class"]=data["Species"].map({"Iris-setosa":0,'Iris-versicolor':1,"Iris-virginica":2}).astype(int)
X=np.array(data.iloc[:,0:4])

Y=np.array(data["class"])

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn import tree

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=1)
model=DecisionTreeClassifier()

model.fit(X_train,Y_train)

prediction=model.predict(X_test)

model.score(X_test,Y_test)
cnf_matrix = confusion_matrix(Y_test, prediction)

print(cnf_matrix)