import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
print(tf.__version__)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    if filenames:
        train_csv_path = os.path.join(dirname, filenames[0])
        test_csv_path = os.path.join(dirname, filenames[1])
        sample_submission_csv_path = os.path.join(dirname, filenames[2])

print(train_csv_path)
print(test_csv_path)
print(sample_submission_csv_path)
df_train = pd.read_csv(train_csv_path)
df_cols = df_train.columns.values.tolist()
print(df_cols)
categorical_columns = ['POSTED_BY','BHK_OR_RK','ADDRESS']
for col in categorical_columns:
    df_cols.remove(col)
Y = df_train[df_cols[-1]].values
X = df_train[df_cols[:-1]].values
X, Y = shuffle(X, Y)
print(X.shape)
print(Y.shape)
num_epoches = 80
batch_size = 128
val_split = 0.15
Xscalar = StandardScaler()
Xscalar.fit(X)

Xtrain = Xscalar.transform(X)
Ytrain = Y
# Yscalar = StandardScaler()
# Yscalar.fit(Y.reshape(-1, 1))

# Ytrain = Yscalar.transform(Y.reshape(-1, 1))
# Ytrain = Ytrain.squeeze()
def classifier1():
    n_features = Xtrain.shape[1]
    inputs = Input(shape=(n_features,))
    x = Dense(512, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
#     x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    
    model.compile(
        loss='mse',
        optimizer='adam'
    )
    history = model.fit(
                    Xtrain,
                    Ytrain,
                    batch_size=batch_size,
                    epochs=num_epoches,
                    validation_split=val_split
                    )
    return history, model
    
def plot_metrics(history):
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    
    loss_train = np.cumsum(loss_train) / np.arange(1,num_epoches+1)
    loss_val = np.cumsum(loss_val) / np.arange(1,num_epoches+1)
    plt.plot(loss_train, 'r', label='Training loss')
    plt.plot(loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
history1, model1 = classifier1()
plot_metrics(history1)
df_test = pd.read_csv(test_csv_path)
df_cols_test = df_test.columns.values.tolist()

categorical_columns = ['POSTED_BY','BHK_OR_RK','ADDRESS']
for col in categorical_columns:
    df_cols_test.remove(col)
df_cols_test
Xtest = df_test[df_cols_test].values
Xtest.shape
Xtest = Xscalar.transform(Xtest)
Ypred = model1.predict(Xtest)
submission_df = pd.read_csv(sample_submission_csv_path)
submission_df['TARGET(PRICE_IN_LACS)'] = Ypred
submission_df.head()
submission_csv_path = '/kaggle/working/submission.csv'
submission_df.to_csv(submission_csv_path)

