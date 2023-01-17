# importing useful libraries



import pandas as pd

import numpy as np

from sklearn.datasets import load_iris
iris=load_iris()
#Using this you can actually go through the data of iris dataset



dir(iris)
iris.feature_names
df=pd.DataFrame(iris.data,columns=iris.feature_names)

df
df['target']=iris.target

df
df['flower_names']=df.target.apply(lambda x:iris.target_names[x])

df
x=df.drop(['target','flower_names'],axis='columns')

x
y=df.target

y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Training the model using KNeighborClassifier



from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier()

model.fit(x_train,y_train)
#Testing the model using test dataset



model.score(x_test,y_test)