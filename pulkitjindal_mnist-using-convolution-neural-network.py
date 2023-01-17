# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("../input/train.csv")
df.info()
df.isnull().sum().sum()
df.head()
y=df['label']
X=df.drop('label',axis=1)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X=X/255
from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization

from keras.optimizers import Adam

from keras import backend as K
batch_size=128

epochs=24

img_rows,img_cols=28,28
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.0001)

X_train=X_train.as_matrix()

X_train=X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
X_val=X_val.as_matrix()

X_val=X_val.reshape(X_val.shape[0],img_rows,img_cols,1)
input_shape=(img_rows,img_cols,1)

import keras

num_classes=10

y_train=keras.utils.to_categorical(y_train,num_classes)

y_val=keras.utils.to_categorical(y_val,num_classes)
model=Sequential()

model.add(Conv2D(64,kernel_size=(5,5),activation='relu',input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=(5,5),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())



model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.6))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
earlystopping=keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,verbose=0,mode='auto')
hist=model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val,y_val),callbacks=[earlystopping])
score=model.evaluate(X_val,y_val,verbose=1)









#from sklearn.decomposition import PCA

#pca_model=PCA(svd_solver='auto',n_components=200)
#X=pca_model.fit_transform(X)

#X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1)
#from sklearn.neural_network import MLPClassifier

#from sklearn.svm import SVC

#from sklearn.linear_model import LogisticRegression
#model=LogisticRegression
#model=SVC(C=1,kernel='rbf',gamma=.001)
#model=MLPClassifier(hidden_layer_sizes=(100,100),verbose=10,max_iter=200,alpha=.01,solver='sgd',tol=.0001,random_state=10,learning_rate_init=.01)
#model.fit(X_train,y_train)
#print(model.score(X_val,y_val))

#print(model.score(X_train,y_train))
from sklearn.metrics import r2_score
y_pred=model.predict(X_val)

y_pred1=model.predict(X_train)
print(r2_score(y_val,y_pred))

print(r2_score(y_train,y_pred1))
df_test=pd.read_csv('../input/test.csv')
X_test=df_test.copy()
X_test=X_test/255

X_test=X_test.as_matrix()

X_test=X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
#X_test=pca_model.fit_transform(X_test)
y_test_pred=model.predict(X_test)
y_out=[np.argmax(y_test_pred[i]) for i in range(y_test_pred.shape[0])]
y_out
submission=pd.DataFrame()
submission['Label']=y_out

submission.index +=1
submission['ImageId']=submission.index

submission=submission[['ImageId','Label']]
submission
submission.to_csv('submission.csv',index=False)