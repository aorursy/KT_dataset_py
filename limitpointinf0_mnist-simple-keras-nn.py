"""
ImageId,Label
1,3
2,7
3,8 
(27997 more lines)
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam
from keras.callbacks import History
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def simple_NN(input_shape, nodes_per=[60], hidden=0, out=2, act_out='softmax', act_hid='relu', drop=True, d_rate=0.1):
  """Generate a keras neural network with arbitrary number of hidden layers, activation functions, dropout rates, etc"""
  model = Sequential()
  #adding first hidden layer with 60 nodes (first value in nodes_per list)
  model.add(Dense(nodes_per[0],activation=act_hid,input_shape=input_shape))
  if drop:
      model.add(Dropout(d_rate))
  try:
    if hidden != 0:
      for i,j in zip(range(hidden), nodes_per[1:]):
          model.add(Dense(j,activation=act_hid))
          if drop:
              model.add(Dropout(d_rate))
    model.add(Dense(out,activation=act_out))
    return(model)
  except:
    print('Error in generating hidden layers')

#Define a function to plot historical data on key statistics of a keras model
def plt_perf(name, p_loss=False, p_acc=False, val=False, size=(15,15), save=False):
  """Plot model statistics for keras models"""
  if p_loss or p_acc:
    if p_loss:
      plt.figure(figsize = size)
      plt.title('Loss')
      plt.plot(name.history['loss'], 'b', label='loss')
      if val:
        plt.plot(name.history['val_loss'], 'r', label='val_loss')
      plt.xlabel('Epochs')
      plt.ylabel('Value')
      plt.legend()
      plt.show()
      if save:
        plt.savefig('loss.png')
    if p_acc:
      plt.figure(figsize = size)
      plt.title('Accuracy')
      plt.plot(name.history['acc'], 'b', label='acc')
      if val:
        plt.plot(name.history['val_acc'], 'r', label='val_acc')
      plt.xlabel('Epochs')
      plt.ylabel('Value')
      plt.legend()
      plt.show()
      if save:
        plt.savefig('acc.png')
  else:
    print('No plotting since all parameters set to false.')
df = pd.read_csv('../input/train.csv')
df_t = pd.read_csv('../input/test.csv')
df.head()
y = to_categorical(np.array(df['label']))
X = np.array(df.drop('label', axis=1))
print(y.shape)
print(X.shape)
model = simple_NN((X.shape[1],), nodes_per=[60,60], hidden=2, out=y.shape[1], drop=True)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=5)                  
history = model.fit(X,y,epochs=1000, validation_split=0.1, callbacks=[early_stopping_monitor], verbose=True)
print('model trained')
plt_perf(history, p_loss=True, p_acc=True, val=True)
preds = model.predict(np.array(df_t))
submit = [np.argmax(x) for x in preds]
print(submit[0:100])
submit = pd.DataFrame.from_records(list(enumerate(submit)), columns=['ImageId','Label'])
submit['ImageId'] = submit['ImageId'] + 1
submit.head()
submit.to_csv('Prediction.csv', index=False)