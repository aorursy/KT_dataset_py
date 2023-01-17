import numpy as np

import pandas as pd



df=pd.read_csv('../input/Iris.csv')

df.head()



X=df.iloc[:,1:5].values

y=df.iloc[:,5].values

y=y.reshape(-1,1)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)



from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le1=LabelEncoder()

le2=LabelEncoder()

y_train=le1.fit_transform(y_train)

y_test=le2.fit_transform(y_test)

y_train=y_train.reshape(-1,1)



ohe=OneHotEncoder()

y_train=ohe.fit_transform(y_train).toarray()



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X__train=sc.fit_transform(X_train)

X__test=sc.fit_transform(X_test)

import keras

from keras.models import Sequential

from keras.layers import Dense

cnn=Sequential()

cnn.add(Dense(output_dim=5,init='uniform',activation='relu',input_dim=4))

cnn.add(Dense(output_dim=5,init='uniform',activation='relu'))

cnn.add(Dense(output_dim=3,init='uniform',activation='softmax'))

cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

cnn.fit(X_train,y_train,batch_size=10,epochs=100)
y_predict=cnn.predict_classes(X_test)
import seaborn as sns

sns.countplot(y_test)

sns.countplot(y_predict)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_predict)

cm
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_predict))