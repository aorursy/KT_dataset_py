import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split #for splitting data
from tensorflow.python import keras 
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt #for plotting
%matplotlib inline
data_file = "/kaggle/input/creditcardfraud/creditcard.csv"
data = pd.read_csv(data_file, delimiter=',')

data.shape 
data.head()

def data_prep(data):
    
    out_y = data.values[:,-1] #just get the last coloumn since that contains the label of fraud/no fraud
    out_x = data.values[:,:-1] #selecting all coloumns except the last one 
    
    return out_x, out_y

x_data, y_data = data_prep(data)

train_x, val_x, train_y, val_y = train_test_split(x_data, y_data, test_size=0.30, random_state=1) #splitting the data for training and validation
#defining the model
model = Sequential()

#adding layers
model.add(Dense(20, input_dim=30, activation='relu')) 
model.add(Dense(20, activation='relu')) 
model.add(Dense(15, activation='relu')) 
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model on our dataset 

history = model.fit(train_x, train_y,epochs=6,validation_data=(val_x, val_y))
def plotLearningCurve(history,epochs):
  epochRange = range(1,epochs+1)
  plt.plot(epochRange,history.history['accuracy'])
  plt.plot(epochRange,history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train','Validation'],loc='upper right')
  plt.show()

  plt.plot(epochRange,history.history['loss'])
  plt.plot(epochRange,history.history['val_loss'])
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train','Validation'],loc='upper right')
  plt.show()
plotLearningCurve(history,6)