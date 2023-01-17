import pandas as pd

from keras.models import Sequential

from keras.layers import Dense
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head()
X = df.drop('Outcome',axis=1)

y = df['Outcome']
model = Sequential()

model.add(Dense(256,input_dim=8,activation='relu'))

model.add(Dense(64,activation='relu'))

model.add(Dense(12,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(X,y,batch_size=140,epochs=1500)
_,accuracy = model.evaluate(X,y,verbose=0)
accuracy*100
predictions = model.predict_classes(X)
predictions