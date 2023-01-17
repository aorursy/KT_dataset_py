# !pip install git+https://github.com/tensorflow/docs
import numpy as np
from numpy import asarray
import pandas as pd
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
df_train.head(15)
embarked = {'S': 2,'C': 1, 'Q': 0} 
df_train['Embarked'] = [embarked[item] for item in df_train['Embarked']]
df_test['Embarked'] = [embarked[item] for item in df_test['Embarked']]
gender = {'male': 1,'female': 0} 
df_train['Sex'] = [gender[item] for item in df_train['Sex']]
df_test['Sex'] = [gender[item] for item in df_test['Sex']]
y_train = df_train.loc[:, df_train.columns == 'Survived']
y_train.head()
X_train = df_train.loc[:, df_train.columns != 'Survived']
X_train = X_train.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)
X_train.head()
type(y_train)
X_train.shape
# X_train = np.asarray(X_train)
# y_train   = np.asarray(y_train)
# X_train.shape
# y_train.shape

# X_train = np.expand_dims(X_train, -1)
# y_train   = np.expand_dims(y_train, -1)
# X_train.shape
# y_train.shape
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(6,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()
model.summary()
# X_train = np.asarray(X_train)
# y_train   = np.asarray(y_train)
# # show_shapes()

# X_train = np.expand_dims(X_train, -1)
# y_train   = np.expand_dims(y_train, -1)
# # show_shapes()
# X_train = X_train.reshape([-1, 891, 11])
X_train.shape
X_train = np.asarray(X_train).astype(np.int32)
EPOCHS = 1000

history = model.fit(
  X_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])
X_test = df_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)
X_test.shape
X_test.head()
prediction = model.predict(X_test)
#create a for loop for prediction
predicted = []
for x in range(0, len(prediction)):
    if prediction[x] < 0.5:
        predicted.append(0)
    else:
        predicted.append(1)
passengerId = df_test['PassengerId']
df = pd.DataFrame({"PassengerId" : passengerId, "Survived" : predicted})
df.to_csv("submission2.csv", index=False)