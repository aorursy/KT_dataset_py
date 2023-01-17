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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
training_data=pd.read_csv('../input/human-activity-recognition-with-smartphones/train.csv')

test_data=pd.read_csv('../input/human-activity-recognition-with-smartphones/test.csv')
print("Training Data: {}".format(training_data.shape))

print("Any Null Values In the Training Data: {}".format(training_data.isnull().values.any()))







print("Test Data: {}".format(test_data.shape))

print("Any Null Values In the Test Data: {}".format(test_data.isnull().values.any()))
X_train=training_data.iloc[:,:-2]

y_train=training_data.iloc[:,-1]



X_test=test_data.iloc[:,:-2]

y_test=test_data.iloc[:,-1]
Category_count=np.array(y_train.value_counts())

activity=sorted(y_train.unique())
plt.figure(figsize=(15,5))

plt.pie(Category_count,labels=activity);
acc=0

gyro=0

others=0

for column in training_data.columns:

    if "Acc" in str(column):

        acc+=1

    elif "Gyro" in str(column):

        gyro+=1

    else:

        others+=1

        
plt.figure(figsize=(12,8))

plt.bar(['Accelerometer','Gyrometer','Others'],[acc,gyro,others],color=['r','g','b']);
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

y_train=encoder.fit_transform(y_train)

y_train=pd.get_dummies(y_train).values
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

y_test=encoder.fit_transform(y_test)

y_test=pd.get_dummies(y_test).values
from sklearn.decomposition import PCA

pca=PCA(n_components=None)

X_train=pca.fit_transform(X_train)

X_test=pca.transform(X_test)

explained_variance=pca.explained_variance_ratio_
explained_variance


from keras.models import Sequential

from keras.layers import Dense,Dropout
model=Sequential()

model.add(Dense(units=64,kernel_initializer='uniform',activation='relu',input_dim=X_train.shape[1]))



model.add(Dense(units=128,kernel_initializer='uniform',activation='relu'))



model.add(Dense(units=64,kernel_initializer='uniform',activation='relu'))



model.add(Dense(units=6,kernel_initializer='uniform',activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,batch_size=256,epochs=22,validation_data=(X_test,y_test))
from pylab import rcParams

rcParams['figure.figsize'] = 10, 4

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred=model.predict(X_test)
y_test_class=np.argmax(y_test,axis=1)

y_pred_class=np.argmax(y_pred,axis=1)
y_test_class
y_pred_class
from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test_class,y_pred_class)

accuracy=accuracy_score(y_test_class,y_pred_class)
cm
accuracy