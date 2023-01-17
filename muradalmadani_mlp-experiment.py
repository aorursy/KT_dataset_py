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
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv("../input/mnist-in-csv/mnist_train.csv") #reading the csv files using pandas
test_data = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
train_data.shape # print the dimension or shape of train data
         
test_data.shape # print the dimension or shape of test data
         
sns.countplot(train_data["label"])
# Plotting a second sample as well as converting into matrix
seven = train_data.iloc[6, 1:]
seven.shape
seven = seven.values.reshape(28, 28)
plt.imshow(seven, cmap='gray')
plt.title("Digit 7")
## Separating the X and Y variable
y = train_data['label'].values.astype('int32')
## Dropping the variable 'label' from X variable
X = train_data.drop(columns = 'label').values.astype('float32')
## Printing the size of data
print("X:", X.shape)
from keras.utils import to_categorical
y_cat = to_categorical(y)
## Normalization
X = X/255.0
# scaling the features
from sklearn.preprocessing import scale
X_scaled = scale(X)
# train test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_cat, test_size = 0.2, train_size = 0.8, random_state = 10)
X_val.shape
from subprocess import check_output
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
y_train.shape
dimData = X_train.shape[1]
nClasses = 10
advanced_act = LeakyReLU(alpha=.003)
model_reg = Sequential()
model_reg.add(Dense(512, activation='linear', input_shape=(dimData,)))
model_reg.add(advanced_act)
model_reg.add(Dropout(0.6))
model_reg.add(Dense(512, activation='linear'))
model_reg.add(advanced_act)
model_reg.add(Dropout(0.6))
model_reg.add(Dense(nClasses, activation='softmax'))

#rmsprop , Adam, SGD
model_reg.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
mlp_reg = model_reg.fit(X_train, y_train, batch_size=256, epochs=20, verbose=2, validation_data=(X_val, y_val))
#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(mlp_reg.history['loss'],'r',linewidth=3.0)
plt.plot(mlp_reg.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(mlp_reg.history['accuracy'],'r',linewidth=3.0)
plt.plot(mlp_reg.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
y_test = test_data['label'].values.astype('int32')
## Dropping the variable 'label' from X variable
X_test = test_data.drop(columns = 'label').values.astype('float32')
X_test=X_test/255
X_test=scale(X_test)
y_test_pred = model_reg.predict_classes(X_test)
y_test.shape
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)
