import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import keras

from keras import optimizers

from keras.models import Sequential, load_model

from keras.layers import InputLayer 

from keras.layers import Dense 

from keras.layers import Dropout 

from keras.constraints import maxnorm

from keras.callbacks import EarlyStopping

from imblearn.over_sampling import SMOTENC

from sklearn.model_selection import train_test_split 

from matplotlib import pyplot as plt

%matplotlib inline
# Read the test data in a pandas DataFrame (Input data files are available in the "../input/" directory)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

data_orig = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')



data = data_orig.reindex(np.random.permutation(data_orig.index))

data.SeniorCitizen.replace([0, 1], ["No", "Yes"], inplace= True)

data.TotalCharges.replace([" "], ["0"], inplace= True)

data.TotalCharges = data.TotalCharges.astype(float)

data.drop("customerID", axis= 1, inplace= True)

data.Churn.replace(["Yes", "No"], [1, 0], inplace= True)



# Now lets apply pandas get_dummies function to one-hot encode all categorical columns:

data = pd.get_dummies(data)



X = data.drop("Churn", axis= 1)

y = data.Churn



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
# Checking if the target is imbalanced.

y.value_counts().plot(kind='bar')
cat_cols = [0]+[i for i in range(3,45)] # categorical columns



sm = SMOTENC(categorical_features=cat_cols, sampling_strategy='minority', random_state=42)

X_smote, y_smote = sm.fit_resample(X, y)
model = Sequential()

model.add(Dense(23, input_dim=46, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(1, activation='sigmoid'))



otim = optimizers.adam(lr=0.0001)



model.compile(loss = "binary_crossentropy", optimizer = otim, metrics=['accuracy'])

history = model.fit(X_smote, y_smote, validation_split=0.2, epochs=2000, verbose=0, batch_size=64)
print("Final validation Accuracy:", round(history.history['val_acc'][-1],2)*100,"%")
plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])

plt.show()
plt.figure(figsize=(10,5))

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('ACCURACY')

plt.xlabel('EPOCH')

plt.ylim((min(history.history['acc'])-0.01,max(history.history['val_acc'])+0.01))

plt.xlim((0,history.epoch[-1]))

plt.legend(['TRAIN', 'TEST'], loc='upper right')

plt.show()
from sklearn.metrics import confusion_matrix



y_pred = model.predict_classes(X_test)



tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(tn, tp, fn, fp)



confusion_matrix(y_test, y_pred)
# summarize model.

model.summary()