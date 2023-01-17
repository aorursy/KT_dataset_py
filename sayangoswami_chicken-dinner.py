import numpy as np
import pandas as pd
import os
import gc
from sklearn.preprocessing import StandardScaler
from IPython.core.debugger import set_trace

print(os.listdir("../input"))
# training set
train = pd.read_csv("../input/train_V2.csv")

# test dataset
test = pd.read_csv("../input/test_V2.csv")

def feature_engineer(df, is_train=True):
    # drop cols that don't matter(?)
    cols_to_drop = ['Id','groupId','matchId', 'matchType']
    df.drop(cols_to_drop, axis=1, inplace=True)
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df = df.fillna(0.0)
    
    print(df.count())
    
    y = None
    
    if is_train:
        y = np.array(df['winPlacePerc'], dtype=np.float64)
        df.drop('winPlacePerc', axis=1, inplace=True)
        
        scaler.fit(df)
    
    X = df    
    X = scaler.transform(X)
    X = np.array(df, dtype=np.float64)  
    
    del df
    gc.collect()
    
    return X, y
 
# preprocessing
scaler = StandardScaler()

X_train, y_train = feature_engineer(train)
X_test, _ = feature_engineer(test, is_train=False)
num_features = X_train.shape[1]
import seaborn as sns

correlations = train.corr()
mask = np.zeros_like(correlations)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(correlations, mask=mask)
# keras 2.2.3 is require for 'restore_best_weights'
# !pip install -qq keras==2.2.3

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

model = Sequential([
    Dense(units=64, activation='selu', input_dim=num_features),
    Dense(units=128, activation='selu'),
    Dropout(0.1),
    Dense(units=64, activation='selu'),
    Dropout(0.1),
    Dense(units=32, activation='selu'),
    Dense(units=16, activation='selu'),
    Dense(units=1, activation='sigmoid')
])

keras.__version__
# hyperparameters
lr = 0.001
epochs = 100
decay = 1e-4
cv_split_ratio = 0.2
batch_size = 20000
patience = 10


optim = keras.optimizers.Adam(lr=lr, decay=decay, amsgrad=True)
model.compile(loss='mean_squared_error', optimizer=optim, metrics=['mae'])

early_stopper = EarlyStopping(monitor='val_mean_absolute_error',
                              min_delta=0,
                              patience=patience,
                              verbose=2,
                              restore_best_weights=True)

history = model.fit(X_train,
                    y_train,
                    validation_split=cv_split_ratio,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopper])
import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

print(history.history.keys())

# Plot training & validation mae values
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Mean Abosulte Error')
plt.ylabel('Mean absolute error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
y_pred = model.predict(X_test, batch_size=10000)

submission = pd.read_csv('../input/sample_submission_V2.csv')
submission['winPlacePerc'] = y_pred

for row in y_pred:
    print(row)
    break

submission.to_csv('submission.csv', index=False)
