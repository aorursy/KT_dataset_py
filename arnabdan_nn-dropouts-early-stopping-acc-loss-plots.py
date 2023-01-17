# Define import



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping

from keras.regularizers import l2

from keras import optimizers



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Import data

train_input = pd.read_csv("../input/train.csv")

test_input = pd.read_csv("../input/test.csv")

train_input.head()
#Training Data input(x) and output(y)

train_x = train_input.drop(['ID_code', 'target'], axis = 1)

train_y = train_input['target']
#Test Data input(x)

test_x = test_input.drop(['ID_code'], axis = 1)
#standardized input

ss = StandardScaler()

train_x_scaled = ss.fit_transform(train_x)

test_x_scaled = ss.transform(test_x)
#Label encoded output

encoder = LabelEncoder()

encoder.fit(train_y)

train_y_encoded = encoder.transform(train_y)
#Definining the NN model

model = Sequential()

model.add(Dense(200, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.001)))

model.add(Dropout(0.4))

model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))

model.add(Dense(1, activation='sigmoid'))
#Defining optimizer

_opt= 'adam'

_loss = 'binary_crossentropy'

#complie model

model.compile(loss=_loss, optimizer=_opt, metrics=['accuracy'])
# Early stopping 

#from keras.callbacks import EarlyStopping

_es_monitor = 'val_loss'

_es_patience = 10

es = EarlyStopping(monitor=_es_monitor, mode='min', verbose=1, patience=_es_patience)
#batch size and number of epchos 

_batch_size = 1

_epochs = 100
#Train model

history = model.fit(train_x_scaled, train_y_encoded, validation_split=0.20,

                    epochs=_epochs, batch_size = len(train_x_scaled), verbose=1, callbacks=[es])
#Evaluate Model's accuracy

metrics = model.evaluate(train_x_scaled, train_y_encoded)

print("\n%s: %.2f%%" % (model.metrics_names[1], metrics[1]*100))
# Plot accuracy - Training vs Validation

import matplotlib.pyplot as plt

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.title('Accuracy - Training vs Validation')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='lower right')

plt.show()

# Plot loss - Training vs Validation

import matplotlib.pyplot as plt



plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='test loss')

plt.title('Loss - Training vs Validation')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()
# Model predict on Test data

predict = model.predict(test_x_scaled)

result = pd.DataFrame({"ID_code": pd.read_csv("../input/test.csv")['ID_code'], "target": predict[:,0]})

print(result.head())



result.to_csv("submission.Arnab.Mar152019.4.csv", index=False)