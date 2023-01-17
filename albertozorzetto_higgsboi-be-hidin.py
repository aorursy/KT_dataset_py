
import numpy as np 
import pandas as pd 
from tensorflow.keras.layers import Input,Dense,Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


data=pd.read_csv('/kaggle/input/higgsb/training.csv')
print(data.isnull().sum())
data
data.info()
#f,ax=plt.subplots(figsize= (256 ,96))  
#sns.heatmap(X.head(200), annot=True, linewidths=.5 ,ax=  ax,square= False)

X = data.drop([ 'Label','EventId'],axis=1,inplace=False)
y = pd.get_dummies(data.Label)
X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.8,shuffle=True,random_state=41)
knn = KNeighborsClassifier(n_neighbors= 5) 
knn.fit(X_train,y_train)
knn.score(X_valid,y_valid)
model=Sequential()
model.add(Input(X.shape[1],batch_size= 32) )
model.add(Dense(64,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.2)) 
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(512,activation="relu"))
model.add(Dense(y.shape [1],activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'] )
model.fit(X.values,y.values,validation_split=0.2,batch_size=32,verbose=1,shuffle = False,epochs =7)