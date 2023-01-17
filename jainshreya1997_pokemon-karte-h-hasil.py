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
train=pd.read_csv("../input/Pokemon.csv", na_values="NA")

from sklearn.model_selection import train_test_split

X=train

y=np.zeros((train.shape[0],1))

y[:,0]=train['Legendary']

X.drop(['Legendary'],axis=1,inplace=True)

X.drop(['Name','#'],axis=1,inplace=True)



X['Type 1']=pd.get_dummies(X['Type 1'])

X['Type 2']=pd.get_dummies(X['Type 2'])

x,x_test,y,y_test=train_test_split(X,y,random_state=0)

print(x_test.shape)

print(x.shape)

"""_dataframe = pd.DataFrame(x,columns=iris_dataset.feature_names)

# create a scatter matrix from the dataframe, color by y_train

grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',

hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)"""

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x,y)

y_pred=knn.predict(x_test)

print("Test set score knn: {:.2f}".format(knn.score(x_test, y_test)))

from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(random_state=0, max_depth=4, learning_rate=0.071)

gbrt.fit(x, y)

print("Accuracy on training set: {:.3f}".format(gbrt.score(x, y)))

print("Accuracy on test set: {:.3f}".format(gbrt.score(x_test, y_test)))