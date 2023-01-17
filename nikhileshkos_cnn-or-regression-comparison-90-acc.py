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

train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
LR_train = train_data.copy()
LR_test = test_data.copy()
CNN_train = train_data.copy()
CNN_test = test_data.copy()
LR_train.head(3)
LR_test.head(3)
print("Training: {} and Test : {}".format(LR_train.shape, LR_test.shape))
train_y = LR_train['label']
# drop the label from train dataset
LR_train.drop(['label'], axis=1, inplace=True)
LR_train.head(2)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(LR_train, train_y, test_size=0.3, random_state=42)
print("X train: {}".format(x_train.shape))
print("Y train: {}".format(y_train.shape))
print("X test: {}".format(x_test.shape))
print("Y test: {}".format(y_test.shape))
from sklearn.linear_model import LogisticRegression
regress = LogisticRegression(max_iter=500)
regress
import warnings
warnings.filterwarnings('ignore')
regress.fit(x_train, y_train)
pred = regress.predict(x_test)
pred
from sklearn.metrics import accuracy_score
regress_acc = accuracy_score(pred, y_test)* 100
print("Regression Score: {}".format(regress_acc))
regress_acc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# reshape for CNN

CNN_train.head(3)
print("CNN Training: {} and CNN Test: {}".format(CNN_train.shape, CNN_test.shape))
CNN_y_train  = CNN_train['label'].values
CNN_x_train = CNN_train.drop(['label'],1).values
CNN_test = CNN_test.values
# let check the values again
CNN_x_train[:5]
CNN_y_train[:5]
CNN_test[:5]
# reshape the values
CNN_x_train = CNN_x_train.reshape(-1,28,28,1)
CNN_test = CNN_test.reshape(-1,28,28,1)
CNN_x_train[:4]
# Do one hot encoding
from keras.utils.np_utils import to_categorical

CNN_y_train[3]
y = to_categorical(CNN_y_train)
print("Label size: {}".format(y.shape))
# split train and validation dataset

CNN_train_split, CNN_test_split, CNN_y_train_split, CNN_y_test_split = train_test_split(CNN_x_train, y, test_size=0.2, random_state=0)
print("\n\nCNN Data Size \n\n")

print("CNN_train_split  size : {}\n".format(CNN_train_split.shape))
print("CNN_test_split  size : {}\n".format(CNN_test_split.shape))
print("CNN_y_train_split size  : {}\n".format(CNN_y_train_split.shape))
print("CNN_y_test_split  size : {}\n".format(CNN_y_test_split.shape))

# Let visualize
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


model.summary()
# Compile

model.compile(optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy'])
batch_size = 64
epochs = 10
history = model.fit(CNN_train_split,CNN_y_train_split, epochs=epochs,
         batch_size=batch_size)
acc = model.evaluate(CNN_test_split, CNN_y_test_split)

# First one is loss and second value in argument

acc_score = round(acc[1] * 100, 2)
print('CNN acc is ', acc_score )


fig, axis = plt.subplots(2, 2, figsize=(12, 14))

for i, ax in enumerate(axis.flat):
    ax.imshow(CNN_test_split[i].reshape(28,28), cmap='binary')
    pred = model.predict(CNN_test_split[i].reshape(1, 28, 28, 1)).argmax()
    real = CNN_y_test_split[i].argmax()
    ax.set_title('Predicted: {} \n Real: {}'.format(pred,real), fontsize=30, color='red')

Logistic_Regression = round(regress_acc, 2)
Logistic_Regression
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Indicator(
    mode = "gauge+number",
    value = Linear_Regression,
    title = {'text': "Logistic Regression Accuracy"},
    domain = {'x': [0, 0.25], 'y': [0, 1]}
))
fig.add_trace(go.Indicator(
    mode = "gauge+number",
    value = acc_score,
    title = {'text': "CNN Accuracy"},
    domain={'x':[0.45,0.80],'y':[0,1]}
))
# fig.update_layout(width=400, height=500)
fig.show()

pred = model.predict_classes(CNN_test, verbose=1)

subm = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
subm.head(2)
subm['Label'] = pred
subm
subm.to_csv("submission.csv",index=False)


