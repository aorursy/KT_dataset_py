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
train=pd.read_csv("/kaggle/input/detecting-anomalies-in-wafer-manufacturing/Train.csv")
test=pd.read_csv("/kaggle/input/detecting-anomalies-in-wafer-manufacturing/Test.csv")
train.head(4)
x=train.drop("Class",axis=1,inplace=False)
y=train["Class"]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(x)
X
from sklearn.decomposition import PCA
pca=PCA(svd_solver="arpack",random_state=42,tol=0.5)
pca.fit(X)
# pca.components_
# pca.explained_variance_ this gives eigen values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier
# m=LogisticRegression()
steps = [("scal",MinMaxScaler()),('pca', PCA(n_components=20)), ('m', LogisticRegression(max_iter=500))]
model=Pipeline(steps=steps)
model.fit(x_train,y_train)
model.score(x_test,y_test)
y_pred=model.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

y
y.value_counts()
from sklearn.utils import resample 
maj=train[train["Class"]==0]
min=train[train["Class"]==1]
mod_min=resample(min,n_samples=1620,replace=True,random_state=42)
mod_min.shape

Train=pd.concat([maj,mod_min])
Train.head()
xs=Train.drop("Class",inplace=False,axis=1)
ys=Train["Class"]
from sklearn.model_selection import train_test_split
xs_train,xs_test,ys_train,ys_test=train_test_split(xs,ys,random_state=42,test_size=0.3)

steps = [("scal",MinMaxScaler()),('pca', PCA(n_components=300)), ('m', LogisticRegression(max_iter=500))]
model=Pipeline(steps=steps)
model.fit(xs_train,ys_train)
model.score(xs_test,ys_test)
ys_pred=model.predict(xs_test)
from sklearn.metrics import classification_report,roc_auc_score
print(classification_report(ys_test,ys_pred))
print(roc_auc_score(ys_test,ys_pred))
stepz = [("scal",MinMaxScaler())]
pipe=Pipeline(steps=stepz)
Test=pipe.fit_transform(test)
from sklearn.model_selection import cross_val_score
cv_scores=cross_val_score(model,xs,ys,cv=10,scoring="roc_auc")
np.mean(cv_scores)
result=pd.DataFrame(model.predict(Test))
submission=result.to_excel("sub1.xlsx",index=False)
steps=[("norm",MinMaxScaler()),("pca",PCA(n_components=300)),("r",RandomForestClassifier(n_estimators=100,random_state=42,max_depth=10,max_features="log2"))]
modelR=Pipeline(steps=steps)
modelR.fit(xs_train,ys_train)
modelR.score(xs_test,ys_test)
ysr_pred=modelR.predict(xs_test)
from sklearn.metrics import classification_report,roc_auc_score
print(roc_auc_score(ys_test,ysr_pred))
print(classification_report(ys_test,ysr_pred))
from sklearn.model_selection import cross_val_score
cv_scores=cross_val_score(modelR,xs,ys,cv=10,scoring="roc_auc")
np.mean(cv_scores)
res=pd.DataFrame(modelR.predict(Test))
submission=res.to_excel("sub3.xlsx",index=False)
steps = [("scal",StandardScaler())]
model=Pipeline(steps=steps)
Xs=model.fit_transform(xs)
stepz = [("scal",StandardScaler())]
pipe=Pipeline(steps=stepz)
Tests=pipe.fit_transform(test)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(Xs,ys,random_state=42,test_size=0.3)
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow.keras.models import Sequential,load_model
import matplotlib
matplotlib.use('agg')
model=Sequential()
model.add(Dense(500,input_shape=(1558,)))
model.add(Activation("relu"))
model.add(Dense(750))
model.add(Activation("relu"))
model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dense(400))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy', metrics=['AUC'], optimizer='adam')
history = model.fit(X_train, y_train,
          batch_size=150, epochs=70,
          verbose=2,
          validation_data=(X_test, y_test)
                   )
loss_and_metrics = model.evaluate(X_test, y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])
ydl_prob=model.predict_classes(X_test)
print(roc_auc_score(y_test,ydl_prob))
model.predict_classes(Test)
dlr=pd.DataFrame(model.predict_classes(Test))
dlr.to_excel("dlr_final.xlsx",index=False)

