!pip install -q kaggle

!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!ls ~/.kaggle

!chmod 600 /root/.kaggle/kaggle.json  # set permission
!kaggle competitions download -c bitsf312-lab1
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import statistics



train = pd.read_csv('/content/train.csv')

train.shape
train.head()
train.info()
train2 = train.dropna()

train2.shape
a = train2.loc[train2['Size'] == '?']

print(a)
train2.loc[3, 'Size'] = 'Small'

train2.loc[225, 'Size'] = 'Small'
a = train2.loc[train2['Number of Special Characters'] == '?']

print(a)
train2.loc[15, 'Number of Special Characters'] = 3

train2.loc[209, 'Number of Special Characters'] = 3

train2.loc[319, 'Number of Special Characters'] = 3

train2.loc[365, 'Number of Special Characters'] = 3
a = train2.loc[train2['Total Number of Words'] == '?']

print(a)
train2.loc[25, 'Total Number of Words'] = 20
a = train2.loc[train2['Number of Quantities'] == '?']

print(a)
train2.loc[61, 'Number of Quantities'] = 2

train2.loc[117, 'Number of Quantities'] = 2

train2.loc[258, 'Number of Quantities'] = 2
a = train2.loc[train2['Number of Insignificant Quantities'] == '?']

print(a)
train2.loc[23, 'Number of Insignificant Quantities'] = 0

train2.loc[172, 'Number of Insignificant Quantities'] = 0

train2.loc[294, 'Number of Insignificant Quantities'] = 0
a = train2.loc[train2['Total Number of Words'] == '?']

print(a)
train2.loc[25, 'Number of Quantities'] = 20
a = train2.loc[train2['Difficulty'] == '?']

print(a)
train2.loc[18, 'Difficulty'] = 2.00

train2.loc[318, 'Difficulty'] = 2.00
pd.to_numeric(train2['Number of Quantities'], errors='coerce')



pd.to_numeric(train2['Number of Insignificant Quantities'], errors='coerce')



pd.to_numeric(train2['Total Number of Words'], errors='coerce')



pd.to_numeric(train2['Number of Special Characters'], errors='coerce')



pd.to_numeric(train2['Difficulty'], errors='coerce')
train2['Size'].unique()
dict1 = {'Small': 0, 'Medium': 1, 'Big': 2}

train2.replace(dict1, inplace=True)

train2.head()
train2 = train2.astype(float)
train2.info()
train2.shape
y = train2['Class']

X = train2.drop(['Class', 'ID'], axis = 1)

print(X.shape, y.shape)
X.shape
corr = X.corr()

corr.style.background_gradient(cmap='coolwarm')
from keras.utils import to_categorical

y2 = to_categorical(y)

y2[:10]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size = 0.2, random_state = 42)
X_train = X_train.reset_index().drop(['index'], axis = 1)

X_test = X_test.reset_index().drop(['index'], axis = 1)

X_train.shape
X_train.head()
from keras.models import Sequential

from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, Flatten

from keras.callbacks import ModelCheckpoint

from keras.callbacks import Callback

from keras.regularizers import l2



model = Sequential([Dense(100, activation='relu', input_dim=11, kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01)),

                    Dense(100, activation = 'relu'),

                    Dense(6, activation = 'softmax') ])



mc = ModelCheckpoint('nnfl_samp_05_reg_final.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size = 5, epochs = 2000, validation_data=(X_test, y_test), callbacks = [mc])
import matplotlib.pyplot as plt



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from keras.models import load_model

#model.save('nnfl_samp_04_reg.h5')

model_saved = load_model('nnfl_samp_05_reg_final.h5')
test = pd.read_csv('/content/test.csv')

test.head()

test.shape
test.replace(dict1, inplace=True)

test.head()
t2 = test.drop(['ID'], axis = 1)

t2.shape
y_fin = model_saved.predict(t2)
y_pred = np.argmax(y_fin, axis = 1)
df_soln = pd.DataFrame(data = [test['ID'], y_pred], index = None)

df = df_soln.transpose()

df['ID'] = df['ID'].astype(int)

df.columns = ['ID','Class']

df['Class'] = df['Class'].astype(int)

df.to_csv('soln_new_last_final_one.csv',index = False)