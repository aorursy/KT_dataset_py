import pandas as pd

import numpy as np



inputPath = "../input/lish-moa/"



train_X = pd.read_csv(inputPath + "train_features.csv")

test_X = pd.read_csv(inputPath + "test_features.csv")

train_y = pd.read_csv(inputPath + "train_targets_scored.csv")
train_X.head()
train_y.head()
test_X.head()
test_X_id = test_X['sig_id']
indexes = train_X.cp_type != 'ctl_vehicle'
train_X = train_X.loc[indexes]
train_y = train_y.loc[indexes]
import matplotlib.pyplot as plt

import seaborn as sns
plt.plot(train_X['g-0'], '.')
sns.distplot(train_X['g-0']);
sns.distplot(train_X['c-0'])
train_X[train_X['c-0'] == -10]
train_X['cp_dose'].value_counts()
plt.hist(np.log(train_X['g-1'] + 10), bins = 100);
train_X_ = train_X.drop(['sig_id', 'cp_type'], axis = 1)

test_X_ = test_X.drop(['sig_id', 'cp_type'], axis = 1)
train_y_ = train_y.drop(['sig_id'], axis = 1)
def lEncoder(value):

    if str(value) == 'D1': return 0

    return 1    
train_X_['cp_dose'] = train_X_['cp_dose'].map(lEncoder)

test_X_['cp_dose'] = test_X_['cp_dose'].map(lEncoder)
train_X_['cp_time'] = train_X_['cp_time'].map({24:-1, 48:0, 72:1})

test_X_['cp_time'] = test_X_['cp_time'].map({24:-1, 48:0, 72:1})
train_X_.head()
def encoder(value):

    ''' As per discussion forum'''

    if value >= 2 or value <= -2:

        return 1

    return 0
for col in train_X_.columns:

    if col != 'cp_time' or col != 'cp_dose':

        train_X_[col] = train_X_[col].map(encoder)

        test_X_[col] = train_X_[col].map(encoder)
train_y_.head()
X, y = train_X_, train_y_
from sklearn.preprocessing import StandardScaler
train_y.isnull().any().sum()
from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 100, test_size = 0.2, shuffle = True)
'''scaler = StandardScaler()

X_train_ = scaler.fit_transform(X_train)

X_val_ = scaler.transform(X_val)''';
from tensorflow import keras
modelPred = keras.Sequential()



modelPred.add(keras.layers.Dense(128, input_shape = (874, )))

modelPred.add(keras.layers.BatchNormalization())

modelPred.add(keras.layers.Dense(2048, activation = 'relu'))

modelPred.add(keras.layers.BatchNormalization())

modelPred.add(keras.layers.Dense(1024, activation = 'relu'))

modelPred.add(keras.layers.Dropout(0.5))

modelPred.add(keras.layers.Dense(1024, activation = 'relu'))

modelPred.add(keras.layers.BatchNormalization())

modelPred.add(keras.layers.Dense(206, activation = 'sigmoid'))



modelPred.summary()
optim = keras.optimizers.Adam(learning_rate=1e-3)

modelPred.compile(optimizer = optim, loss = keras.losses.CategoricalCrossentropy())
X_train.shape, y_train.shape, X_val.shape, y_val.shape
history = modelPred.fit(

    X_train,

    y_train,

    batch_size = 32,

    epochs = 5,

    validation_data=(X_val, y_val)

)
history = modelPred.fit(

    X_train,

    y_train,

    batch_size = 64,

    epochs = 2,

    validation_data=(X_val, y_val)

)
y_predictions = modelPred.predict(test_X_)
y_predictions.shape
test_X_id.values.shape
final = np.hstack([test_X_id.values.reshape(-1, 1), y_predictions])
final.shape
final.shape



test_X.columns.shape



submission = pd.DataFrame(final, columns = train_y.columns)



submission.head()



submission.to_csv('submission.csv', index = False)