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
df = pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

df.head()
X = df.drop(columns = ['label']).values

y = df['label']
y.min()
df['label'].unique()
X
X =X.reshape(-1,28,28)
X = X /255
X.max()
from matplotlib.pyplot import imshow

imshow(X[0])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y)
X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

y_train=label.fit_transform(y_train)

y_test=label.transform(y_test)
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
import tensorflow as tf

tf.__version__
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,BatchNormalization,Dropout
model = Sequential()

model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))

model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Flatten())

model.add(Dense(units = 512 , activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(units = 24 , activation = 'softmax'))

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()
X_train.shape
y_test.shape,X_test.shape
from tensorflow.keras.callbacks import EarlyStopping

stop=EarlyStopping(patience=3,monitor='val_accuracy')
model.fit(X_train,y_train,epochs = 100,validation_data=(X_test,y_test),callbacks=[stop])
test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
test.head()
X_test = test.drop(columns= ['label'])

y_test = test['label']
X_test = X_test/255
X_test.max()
X_test = np.array(X_test)
X_tset = X_test.reshape(-1,28,28,1)
X_tset.shape
y_preds = model.predict_classes(X_tset)
y_preds=label.inverse_transform(y_preds)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true =y_test,y_pred=y_preds )
accuracy