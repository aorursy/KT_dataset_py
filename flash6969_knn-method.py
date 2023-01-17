# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Any results you write to the current directory are saved as output.

df= pd.read_csv('../input/Iris.csv')

df.Species.replace(to_replace='Iris-setosa',value=0,inplace=True)

df.Species.replace(to_replace='Iris-versicolor',value=1,inplace=True)

df.Species.replace(to_replace='Iris-virginica',value=2,inplace=True)

df



X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y = df['Species']





from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)



from sklearn.neighbors import KNeighborsClassifier



knn= KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train.values,y_train.values)



knn.predict(X_test.values)



knn.score(X_test,y_test)
