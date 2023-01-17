# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filename = "../input/datasetandroidpermissions/train.csv"

df = pd.read_csv(filename,sep=";")

df.head()
#number of benign and malicious

df.type.value_counts()
#malicious

pd.Series.sort_values(df[df.type==1].sum(), ascending=False)[1:11]
#benign

pd.Series.sort_values(df[df.type==0].sum(), ascending=False)[:10]
import seaborn as sns

import matplotlib.pyplot as plt

benign=pd.Series.sort_values(df[df.type==0].sum(), ascending=False)[:10]

malicious=pd.Series.sort_values(df[df.type==1].sum(), ascending=False)[1:11]
plt.figure(figsize=(10,7))

sns.barplot(x=benign.index,y=benign)

plt.figure(figsize=(10,7))

sns.barplot(x=malicious.index,y=malicious)
y=df.type

X=df.drop('type',axis=1)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test= train_test_split(X,y,test_size=0.2,random_state=0)

Y_train.shape
import keras

from keras.models import Sequential

from keras.layers import Dense
#creating input layer and first hidden layer

classifier = Sequential()

classifier.add(Dense(6, input_shape=(12,),activation='relu',kernel_initializer='uniform'))



#creating multiple hidden layers

classifier.add(Dense(6, activation='relu',kernel_initializer='uniform'))





#Output layer

classifier.add(Dense(1, activation='sigmoid',kernel_initializer='uniform'))

   



#Compiling the ANN

classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])



#Fitting the ANN to the training set

classifier.fit(X_train,Y_train,batch_size=10)



y_pred=classifier.predict(X_test)

y_pred=(y_pred > 0.5)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test,y_pred)
#Y_train,Y_test=

Y_train_reshaped=np.array(Y_train).reshape(-1,1)

Y_test_reshaped=np.array(Y_test).reshape(-1,1)
#standardization

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
import tensorflow as tf

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer= 'adam', loss='binary_crossentropy' ,metrics= ['accuracy'])

ann.fit(X_train,Y_train_reshaped,batch_size = 32,epochs=100)
Y_pred = ann.predict(X_test)

Y_pred = (Y_pred > 0.5)

print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test_reshaped.reshape(len(Y_test_reshaped),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(Y_test, Y_pred)

print(cm)

accuracy_score(Y_test, Y_pred)