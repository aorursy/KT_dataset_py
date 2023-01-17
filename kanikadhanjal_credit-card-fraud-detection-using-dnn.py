# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# First, we'll import helpful packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# read the data set
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
# Determine number of fraud and genuine cases in dataset
print(len(df[df['Class'] == 1]))
print(len(df[df['Class'] == 0]))
#finding duplicate rows in dataset
duplicateRowsDF = df[df.duplicated()]

print("Duplicate Rows except first occurrence based on all columns are :")
print(duplicateRowsDF.head())

print(duplicateRowsDF.shape)

# Number of duplicate fraud cases
print(len(duplicateRowsDF[duplicateRowsDF["Class"] == 1]))

# Number of duplicate genuine cases
print(len(duplicateRowsDF[duplicateRowsDF["Class"] == 0]))

print(duplicateRowsDF.index)

##Removing duplicates from the dataset
df = df.drop(duplicateRowsDF.index)

##after removing duplicate rows, printing the shape of the dataset
print(df.shape)
# Determine number of fraud and genuine cases in dataset
Fraud = df[df['Class'] == 1]
Valid = df[df['Class'] == 0]
print(len(df[df['Class'] == 1]))
print(len(df[df['Class'] == 0]))
print("Length of dataset after removing duplicates is", len(df))
print(df.shape)
fraud_perc = len(Fraud) / float(len(df))
print(fraud_perc)
# Since most of data has already been scaled, scaling two columns that are left to scale (Amount and Time)
from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)
Amount = df['scaled_amount']
Time = df['scaled_time']
df.insert(0, 'Amount', Amount)
df.insert(1, 'Time', Time)
df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

# Amount and Time are Scaled!
print(df.head())
# dividing the X and the Y from the dataset
X = df.drop(['Class'], axis = 1)

Y = df["Class"]
print(X.shape)
print(Y.shape)

# getting just the values for the sake of processing

xData = X.values
yData = Y.values
#Split the data into train and testing
X_train, X_test, Y_train, Y_test = train_test_split(xData, yData, test_size=0.1, random_state=42, stratify=Y)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(X_val.shape, Y_val.shape)
batch_size = 512

#Build Deep neural networks
n_cols_2 = xData.shape[1]

#create model
model = Sequential()

#add layers to model
model.add(Dense(16, activation='relu', input_shape=(n_cols_2,)))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

#Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')
#fiiting the model
history=model.fit(X_train, Y_train, batch_size = batch_size, epochs = 100, validation_data= (X_val, Y_val), verbose=1)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
#print(y_pred)
#Let's see how our model performed
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))
print("Confusion Matrix\n",confusion_matrix(Y_test,y_pred))
print("\n")