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
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print(test)
train.shape,test.shape
train.head()
test.head()
sns.countplot(x = 'label' , data = train)
train.isnull().values.any()
for i in range(2):
    img = np.array(train.loc[i+100][1:]).reshape(28,28)
    label = train.loc[i+100][0]
    plt.imshow(img)
    plt.show()
    print('the label is ',label)
train_y = train['label']
train.pop('label')
train_x = train
train_x.head()
train_y.head()
for col in train_x.columns:
    train_x[col] = train_x[col]/255
for col in test.columns:
    test[col] = test[col]/255
print(test)
test.isnull().values.any()
def encode_label(labels,numclasses):
    result = np.zeros((int(len(labels)),int(numclasses)))
    for i,label in enumerate(labels):
        result[i,label] = 1
    return result

encoded_train_y = encode_label(train_y,10)
print(encoded_train_y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(train_x,encoded_train_y,test_size = 0.3,random_state = 42)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
X_train.isnull().values.any()
X_train = np.array(X_train)
X_test = np.array(X_test)
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()

classifier.add(Dense(units = 16 , activation = 'relu' , input_dim = 784))
classifier.add(Dense(units = 8 , activation = 'relu'))
classifier.add(Dense(units = 8 , activation = 'relu'))
classifier.add(Dense(units = 10 , activation = 'sigmoid'))
import keras
opt = keras.optimizers.Adam(learning_rate=0.001)
classifier.compile(loss='categorical_crossentropy', optimizer=opt)
classifier.fit(X_train,Y_train,batch_size = 5,epochs= 60)
y_pred = classifier.predict(X_test)
print(y_pred[0:2])
prediction = np.argmax(y_pred,axis = 1)
print(prediction[:20])
Y_test = np.argmax(Y_test,axis = 1)
print(Y_test[:20])
correct = 0
wrong = 0
total = 0
for i in range(len(prediction)):
    total += 1
    if(Y_test[i] == prediction[i]):
        correct += 1
    else:
        wrong += 1
print('correct percentage :',(correct/total)*100)
print('wrong percentage : ',(wrong/total)*100)
test_arr = np.array(test)
test_pred = classifier.predict(test_arr)
print(test_pred[:5])
test_pred_labels = np.argmax(test_pred,axis = 1)
print(test_pred_labels[:6])
print(type(test_pred_labels))
import csv
count = 0
with open('submission.csv','w',newline ='') as csvfile:
    fieldnames = ['ImageId','Label']
    thewriter = csv.DictWriter(csvfile,fieldnames = fieldnames)
    
    thewriter.writeheader()
    
    for i in test_pred_labels:
        count += 1
        thewriter.writerow({'ImageId': count,'Label':i})