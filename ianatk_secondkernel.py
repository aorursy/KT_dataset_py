import numpy as np

import pandas as pd



data = pd.read_csv('../input/train.csv')

X = data.drop(columns=['label']).values

y = pd.get_dummies(data.label.astype(str)).values
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)



from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

X_train = scale.fit_transform(X_train.astype(float))

X_test = scale.transform(X_test.astype(float))



from keras.models import Sequential

from keras.layers import Dense, Activation



model = Sequential()

model.add(Dense(X.shape[1]//2, input_dim=X.shape[1],activation='relu'))

model.add(Dense(X.shape[1]//2,activation='relu'))

model.add(Dense(X.shape[1]//4,activation='relu'))

model.add(Dense(y.shape[1], activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



model.fit(X_train, y_train, epochs=10, batch_size=32)



y_pred = model.predict(X_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))

print(cm)

print('Accuracy: ,',cm.trace()/cm.sum())