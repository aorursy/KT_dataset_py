import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense,Flatten,Dropout,Input
from tensorflow.python.keras.models import Sequential

data=pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')
data
X = data.drop('target_class', axis=1 ,inplace=False)
y = pd.get_dummies(data['target_class']) 
print(X.shape)
y
model=Sequential()
model.add(Input(X.shape[1]))
model.add(Dense(16,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(3,activation="relu"))
model.add(Dense(y.shape [1],activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'] )
model.fit(X,y,validation_split=0.2,verbose=1,shuffle=True,epochs=2)