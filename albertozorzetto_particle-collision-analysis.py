import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense,Flatten,Dropout,Input
from tensorflow.python.keras.models import Sequential
data = pd.read_csv( "/kaggle/input/particle-identification-from-detector-responses/pid-5M.csv")
data
print(data.shape)
print(np.unique(data['id']))
data.isnull().sum()
X = data.drop('id',axis=1 ,inplace=False)
y=pd.get_dummies(data['id'])
lr=0.05
epochs=3
model=Sequential()
model.add(Input(X.shape[1]))
model.add(Dense(32,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64,activation="relu"))
model.add(Dense(4,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'] )
model.fit(X,y,validation_split=0.2,verbose=1,shuffle=True,epochs=epochs)