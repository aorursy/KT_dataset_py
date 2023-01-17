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
df=pd.read_csv("/kaggle/input/bank-note-authentication-uci-data/BankNote_Authentication.csv")
df.head()
df.describe()
# Count the number of observations of each class

print('Observations per class: \n', df['class'].value_counts())
# Import seaborn

import seaborn as sns

#import matplotlib

import matplotlib.pyplot as plt

# Use pairplot and set the hue to be our class

sns.pairplot(df, hue='class') 



# Show the plot

plt.show()
# Import the sequential model and dense layer

from keras.models import Sequential

from keras.layers import Dense



# Create a sequential model

model = Sequential()

# Add a dense layer 

model.add(Dense(1, input_shape=(4,), activation='sigmoid'))



# Compile your model

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])



# Display a summary of your model

model.summary()
#defining features and target variable

y = df['class']

X = df.drop(columns = ['class'])

from sklearn.model_selection import train_test_split

#splitting the data into train and test set 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Train your model for 20 epochs

model.fit(X_train, y_train, epochs=20)



# Evaluate your model accuracy on the test set

accuracy = model.evaluate(X_test, y_test)[1]



# Print accuracy

print('Accuracy:',accuracy)
# Prediction

preds = model.predict(X_test)



for pred in enumerate(preds):

  print("{} | {}".format(pred,y_test))
# Import the early stopping callback

from keras.callbacks import EarlyStopping



# Define a callback to monitor val_acc

monitor_val_acc = EarlyStopping(monitor="val_acc", 

                       patience=5)



# Train your model using the early stopping callback

model.fit(X_train, y_train, 

           epochs=1000, validation_data=(X_test,y_test),

           callbacks=[monitor_val_acc])