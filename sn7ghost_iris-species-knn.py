# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.neighbors import KNeighborsRegressor

from sklearn import metrics



from sklearn.model_selection import train_test_split

# from sklearn.model_selection import train_test_split



# Any results you write to the current directory are saved as output.
#Loading the data

df=pd.read_csv('../input/Iris.csv')

df=df.set_index('Id')

df.head(5)

df['Species'].unique()

df['Species']=df['Species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
#Spliting up the data:

from sklearn.model_selection import train_test_split

X=(df.drop(['Species'],1))

y=(df['Species'])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

knn=KNeighborsRegressor(n_neighbors=2)

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print(type(y_pred))
y_pred

#Calculating accuracy

l=len(y_pred)

t=np.count_nonzero(y_pred-y_test)

accu=(l-t)/l

print(accu)
metrics.confusion_matrix(y_test,y_pred)
#