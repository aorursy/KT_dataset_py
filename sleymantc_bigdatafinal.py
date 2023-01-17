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
Data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

print("minimum item_cnt_day:")

print(Data.item_cnt_day.min())

print("minimum item_price:")

print(Data.item_price.min())

print("minimum item_id:")

print(Data.item_id.min())

print("minimum shop_id:")

print(Data.shop_id.min())

print("minimum date_block_num:")

print(Data.date_block_num.min())

Data = Data[Data['item_cnt_day']>0]

Data = Data[Data['item_price']>0]

print("after cleaning minimum item_cnt_day:")

print(Data.item_cnt_day.min())

print("after cleaning minimum item_price:")

print(Data.item_price.min())
Data = Data[['date_block_num','shop_id','item_id','item_price','item_cnt_day']]

from sklearn.model_selection import train_test_split

X = Data.iloc[:,:4]

y = Data.iloc[:,4:5]



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(12, input_dim=4, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=3, batch_size=10)

_, accuracy = model.evaluate(X_train, y_train)

print('Accuracy: %.2f' % (accuracy*100))
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score

a = accuracy_score(y_pred,y_test)

print('Accuracy is:', a*100)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()
plt.plot(history.history['loss']) 

plt.plot(history.history['val_loss']) 

plt.title('Model loss') 

plt.ylabel('Loss') 

plt.xlabel('Epoch') 

plt.legend(['Train', 'Test'], loc='upper left') 

plt.show()