from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn import tree

import pandas as pd
iris=load_iris()

iris
df=pd.DataFrame(iris.data,columns=iris.feature_names)

df['class']=iris.target
x=df.iloc[:,:-1].values



y=df.iloc[:,-1:].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
sc=StandardScaler()

sc_x=sc.fit(x_train)

x_train=sc_x.transform(x_train)
pca=PCA(n_components=2)
pca.fit(x_train)
x_train=pca.transform(x_train)
classifier=LogisticRegression(random_state=42)
classifier.fit(x_train,y_train)
x_test=sc_x.transform(x_test)



x_test=pca.transform(x_test)



y_pred=classifier.predict(x_test)



from sklearn.metrics import accuracy_score



accuracy_score(y_test,y_pred)
from sklearn.pipeline import Pipeline
pipe_lr=Pipeline([('scl',StandardScaler()),

                 ('pca',PCA(n_components=2)),

                  ('clf',LogisticRegression(random_state=42))

                 ])
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3,random_state=42)
pipe_lr.fit(X_train,y_train)
pipe_lr.fit(X_train,y_train)
y_pred=pipe_lr.predict(X_test)



accuracy_score(y_test,y_pred)