# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip')

train_data.head()
test_data=pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip')
train_data.shape
train_data.info()
numerical = ['bone_length','rotting_flesh','hair_length','has_soul']

categorical = ['color','type']
corr = train_data.corr()

#heatmap gives a visual representation of correlation between different attributes of the data

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
sns.pairplot(train_data,hue='type')
import matplotlib.pyplot as plt
#Check if numerical attributes have normal distributions

train_data[numerical].hist(bins=15, figsize=(15, 6), layout=(2, 4),rwidth=0.9,grid=False,color='purple');
#Visualize categorical variables

sns.set()

sns.countplot(train_data['color'])
sns.set()

sns.countplot(train_data['type'])
#to plot countplots simultaneously

#fig, ax = plt.subplots(2, figsize=(6, 6))

#for variable, subplot in zip(categorical, ax.flatten()):

    #sns.countplot(train_data[variable], ax=subplot)
test_data.shape
test_data.head()
test_data.info()
train_data['color'].unique()
#Check if train and test data have the same categories

test_data['color'].unique()
#one-hot-encoding categorrical attribute:color

train_data=pd.concat([train_data,pd.get_dummies(train_data['color'])],axis=1)

train_data.drop('color',axis=1,inplace=True)

train_data.head()
test_data=pd.concat([test_data,pd.get_dummies(test_data['color'])],axis=1)

test_data.drop('color',axis=1,inplace=True)

test_data.head()
X=train_data.drop(['id','type'],axis=1)

y=pd.get_dummies(train_data['type'])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
from keras.layers import Dense

from keras.models import Sequential
model = Sequential()

model.add(Dense(100,input_shape=(X.shape[1],)))

model.add(Dense(100,activation='relu'))

model.add(Dense(100,activation='relu'))

model.add(Dense(3,activation='softmax'))

model.summary()
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
train=model.fit(x=X_train,y=y_train,batch_size=10,epochs=10,verbose=2,validation_data=(X_test,y_test))
plt.figure(figsize=(5,5))

plt.plot(train.history['accuracy'],'r',label='Training accuracy')

plt.plot(train.history['val_accuracy'],'b',label='Validation accuracy')

plt.legend()
pred=model.predict(test_data.drop('id',axis=1))
pred_final=[np.argmax(i) for i in pred]
submission = pd.DataFrame({'id':test_data['id'], 'type':pred_final})

submission.head()
submission.to_csv('../working/submission.csv', index=False)