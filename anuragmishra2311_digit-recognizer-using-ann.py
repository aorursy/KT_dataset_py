import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Flatten
import pydot
import scikitplot as skplt
df=pd.read_csv('../input/digit-recognizer/train.csv')
df.head()
X=df.iloc[:,1:]
y=df.iloc[:,0]
X.shape,y.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=1)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
X_train=X_train/255
X_test=X_test/255
model=keras.models.Sequential()
model.add(Dense(units = 128, kernel_initializer = 'he_uniform',activation='relu',input_dim = 784))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer = 'Adamax', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()
keras.utils.plot_model(model)
model_history=model.fit(X_train, y_train,validation_split=0.1,epochs = 25)
pd.DataFrame(model_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
model.evaluate(X_test,y_test)
y_pred=np.argmax(model.predict(X_test).round(2),axis=1)
y_pred
skplt.metrics.plot_confusion_matrix(y_test,y_pred,figsize=(20,20))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))