import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix


df=pd.read_csv("../input/iris/Iris.csv")
from sklearn.datasets import load_iris
iris=load_iris()
for keys in iris.keys() :
    print(keys)
X=iris.data
y=iris.target
print("Target names: {}".format(iris['target_names']))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
print("X_train shape {}" .format(X_train.shape))
print("y_train shape {}" .format(y_train.shape))

#iris_dataframe = pd.DataFrame(X_train, columns=df.columns)
# create a scatter matrix from the dataframe, color by y_train
#grr = pd.scatter_matrix(df, c=y_train, figsize=(15, 15), marker='o',
#hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

pd.plotting.scatter_matrix(df, alpha=0.8,c='y', figsize=(20, 20), marker='o',hist_kwds={'bins': 20}, s=60, cmap = 'tab20c')

plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
X_new=np.array([[5, 2.9, 1, 0.2]])
print('newshape={}' .format(X_new.shape))
prediction=knn.predict(X_new)
print('prediction {}' .format(prediction))
print("Predicted target name: {}".format(iris['target_names'][prediction]))
y_pred=knn.predict(X_test)
print("test set prediction {}" .format(y_pred))
print('test score {:.2f}' .format(np.mean(y_pred==y_test)))
print('model test score {:.2f}' .format(knn.score(X_test,y_test)))
