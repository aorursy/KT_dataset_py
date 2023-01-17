import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('../input/diabetes.csv')
data.head()
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(7)
data.info()
#dataset = np.loadtxt('H:/pima-indians-diabetes-database/diabetes.csv', delimiter=',')
X = data.iloc[:,0:8]
Y = data.iloc[:,8]

pd.DataFrame(X).head()
pd.DataFrame(Y).head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
model = Sequential()
model.add(Dense(12 , input_dim =8 , activation='relu'))
model.add(Dense(8 , activation='relu'))
model.add(Dense(1 , activation='sigmoid'))

model.summary()
model.compile(loss = 'binary_crossentropy' , optimizer='adam' ,metrics=['accuracy'])
model.fit(X_train , y_train , validation_split=0.33 , epochs=200 , batch_size=10)
y_pred = model.predict_classes(X_test , batch_size=10, verbose=0)
from sklearn import metrics
acc = metrics.accuracy_score(y_test , y_pred)
acc
