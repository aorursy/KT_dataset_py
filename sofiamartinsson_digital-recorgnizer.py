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
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df.head()
X = df.drop('label', axis=1).values
y = df['label'].values
#reshape data to fit for images
X = X.reshape(42000,28,28,1)
X = X/255
#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
##create categorical values for my output
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
###Create model and train it
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Compile and train model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(x=X_train, 
          y=y_train_cat, 
          epochs=10, 
          validation_data=(X_test, y_test_cat), 
          callbacks=[early_stop])

evaluate = pd.DataFrame(model.history.history)
#loss
evaluate[['loss', 'val_loss']].plot()
plt.show()
#accuracy
evaluate[['accuracy', 'val_accuracy']].plot()
plt.show()
#predict
y_pred = model.predict_classes(X_test)

#evaluate
classification = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(classification)
print(cm)
##Run with test data
df_validate = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
X_valid = df_validate.to_numpy()
X_valid = X_valid.reshape(28000,28,28,1)
X_valid = X_valid/255
###Predict
prediction = model.predict_classes(X_valid)
#print to file
output = pd.DataFrame({'ImageId': list(df_validate.index.values+1) , 'Label': prediction})
output.to_csv('my_submission.csv', index=False)

