import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense,Embedding,SimpleRNN,Dense,Dropout
from keras.optimizers import Adam
from keras import metrics, regularizers
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.columns=df.columns.str.lower()
df.describe()
X=df.drop(columns='outcome')
y=df.outcome
X
X[X.values==0]
zero_values=SimpleImputer(missing_values=0,strategy="mean")
X=zero_values.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=234)
ax=sns.countplot(x='outcome',data=df,palette="Set3")
activation_func='selu'
validation_split_=0.1
regularizers_lr2=0.01
verbose_=1
epochs_=32
batch_size_=16
size=10000
input_shape=8
kernel_initializer_='random_uniform'
adam = Adam(lr = 0.01)

model=Sequential()
model.add(Embedding(size,input_shape, trainable=True,input_length=input_shape))
model.add(SimpleRNN(16,activation=activation_func,kernel_initializer=kernel_initializer_,kernel_regularizer=regularizers.l2(regularizers_lr2),return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(8,activation=activation_func,kernel_initializer=kernel_initializer_,kernel_regularizer=regularizers.l2(regularizers_lr2)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
model.summary()
history=model.fit(X_train, y_train,
          epochs=epochs_, 
          batch_size=batch_size_, 
          verbose=verbose_,
          validation_split=validation_split_)

predicted=model.predict(X_test)
predicted[:3]
results = confusion_matrix(y_test, predicted.round())
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(y_test, predicted.round()) )
print ('Report : ')
print (classification_report(y_test, predicted.round()) )
fig, ax = plt.subplots(figsize=(10, 7))
plt.plot(history.history['accuracy'], label='Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.ylabel('Acc')
plt.xlabel('Epoch Sayısı')
plt.legend(loc="upper left")
plt.show()
