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
import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
Train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
Test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
Train_data.shape
Train_data.head()
X=Train_data.drop('label',axis=1).values
y=Train_data['label'].values
X.shape
test = Test_data.values
X=X/255
test=test/255
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
print(X_train.shape)
print(X_test.shape)
X_train  = X_train.reshape(-1,28,28)
X_test = X_test.reshape(-1,28,28)
plt.matshow(X_train[0])
y_train[0]
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation

model = Sequential()
model.add(Flatten(input_shape=[28,28]))
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
model.fit(X_train,y_train,epochs=35)
model.evaluate(X_test,y_test)
test=test.reshape(-1,28,28)
yp=model.predict(test)
plt.matshow(test[0])
yp[0]
y_pred=np.argmax(yp, axis=1)
y_pred
plt.matshow(test[0])
y_pred[0]
ypredictions = model.predict(X_test)
ypredictions=np.argmax(ypredictions, axis=1)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,ypredictions)
print(cm)
import seaborn as sns
plt.subplots(figsize=(15,8))
sns.heatmap(cm,annot=True,fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')
sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
sub['Label']=y_pred
#sub.to_csv("Submission.csv",index=False)
sub.head()