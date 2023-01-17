# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings("ignore")



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





from sklearn.metrics import confusion_matrix, cohen_kappa_score

from sklearn.metrics import f1_score, recall_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers

import tensorflow as tf





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

data.head()
print(data.shape)

print(data.columns)
data.isnull().sum().any()
data.Class.value_counts().rename(index = {0:'Not Fraud', 1:'Fraud'})
data['Time'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1, 1))

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
train_x, test_x = train_test_split(data,test_size = 0.3,random_state=42)

train_x = train_x[train_x.Class == 0] 

train_x = train_x.drop(['Class'], axis=1) 





test_y = test_x['Class']

test_x = test_x.drop(['Class'], axis=1)
input_dim = train_x.shape[1]

encoding_dim = int(input_dim / 2) - 1

hidden_dim = int(encoding_dim / 2)

learning_rate = 1e-7



input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)

encoder = Dense(hidden_dim, activation="relu")(encoder)

decoder = Dense(hidden_dim, activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

nb_epoch = 100

batch_size = 128
autoencoder.compile(metrics=['accuracy'],

                    loss='mean_squared_error',

                    optimizer='adam')



cp = ModelCheckpoint(filepath="autoencoder_fraud.h5",

                               save_best_only=True,

                               verbose=0)



tb = TensorBoard(log_dir='./logs',

                histogram_freq=0,

                write_graph=True,

                write_images=True)



history = autoencoder.fit(train_x, train_x,

                    epochs=nb_epoch,

                    batch_size=batch_size,

                    shuffle=True,

                    validation_data=(test_x, test_x),

                    verbose=1,

                    callbacks=[cp, tb]).history
autoencoder = load_model('autoencoder_fraud.h5')
plt.plot(history['loss'], linewidth=2, label='Train')

plt.plot(history['val_loss'], linewidth=2, label='Test')

plt.legend(loc='upper right')

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()
pred = autoencoder.predict(test_x)
mse = np.mean(np.power(test_x - pred, 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,

                        'True_class': test_y})
error_df.Reconstruction_error.values
threshold_fixed = 5

pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

matrix = confusion_matrix(error_df.True_class, pred_y)
tpos = matrix[0][0]

fneg = matrix[1][1]

fpos = matrix[0][1]

tneg = matrix[1][0]
print( 'Accuracy: '+ str(np.round(100*float(tpos+fneg)/float(tpos+fneg + fpos + tneg),2))+'%')

print( 'Cohen Kappa: '+ str(np.round(cohen_kappa_score(error_df.True_class, pred_y),3)))

print("Sensitivity/Recall for Model : {}".format(round(recall_score(error_df.True_class, pred_y), 2)))

print("F1 Score for Model : {}".format(round(f1_score(error_df.True_class, pred_y), 2)))