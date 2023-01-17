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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")
df.head()
df.isna().sum()
plt.figure(figsize=(8,8))
sns.countplot(x='Exited', data=df)
plt.xlabel('0: Customers still with the bank, 1: Customers exited the bank')
plt.ylabel('Count')
plt.title('Bank Customers Churn Visualization')
plt.show()
df.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)
df.drop(['Geography', 'Gender'], axis=1, inplace=True)
df.columns
X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
ann = Sequential()
ann.add(Dense(units=30, kernel_initializer='he_uniform', activation='relu'))
ann.add(Dense(units=10, kernel_initializer='he_uniform', activation='relu'))
ann.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
batch_size=32
EPOCHS=35

history = ann.fit(X_train, y_train.values, batch_size=batch_size, validation_split=0.33, epochs=EPOCHS)
ann.summary()
plt.figure(figsize=(8,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred[:5]
cm = confusion_matrix(y_test, y_pred)
cm
score = accuracy_score(y_pred,y_test)
print('The accuracy for ANN model is: {}%'.format(score*100))
def predict_exit(sample_value):
  sample_value = np.array(sample_value)
  sample_value = sample_value.reshape(1, -1)
  sample_value = sc.transform(sample_value)

  return ann.predict(sample_value)
# Predictions
# Value order 'CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary'.
sample_value = [738, 62, 10, 83008.31, 1, 1, 1, 42766.03]
if predict_exit(sample_value)>0.5:
  print('Prediction: High chance of exiting!')
else:
  print('Prediction: Low chance of exiting.')