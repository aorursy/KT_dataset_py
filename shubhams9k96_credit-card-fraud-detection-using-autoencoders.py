# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))

sns.set_style('dark')

sns.set_context('talk')

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/creditcard.csv', encoding = 'latin1')
df.info()
df.shape
df.isnull().values.any()

print(("Distribution of fraudulent points: {:.2f}%".format(len(df[df['Class']==1])/len(df)*100)))

sns.countplot(df['Class'])

plt.title('Class Distribution')

plt.xticks(range(2),['Normal','Fraud'])

plt.show()

normal = df[df['Class']==0]

fraud = df[df['Class']==1]

print("Normal datapoints: ", normal.shape[0])

print("Fraud datapoints: ", fraud.shape[0])
normal['Amount'].describe()
fraud['Amount'].describe()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize = (10,10) )

f.suptitle('Amount per transaction by class')



bins = 10



ax1.hist(fraud.Amount, bins = bins)

ax1.set_title('Fraud')



ax2.hist(normal.Amount, bins = bins)

ax2.set_title('Normal')



ax1.grid()

ax2.grid()

plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

plt.xlim((0, 20000))

plt.yscale('log')

plt.show();
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,10))

f.suptitle('Time of transaction vs Amount by class')



ax1.scatter(fraud.Time, fraud.Amount, marker='.')

ax1.set_title('Fraud')

ax1.grid()

ax2.scatter(normal.Time, normal.Amount, marker='.')

ax2.set_title('Normal')

ax2.grid()

plt.xlabel('Time (in Seconds)')

plt.ylabel('Amount')

plt.show()
data = df.drop(['Time'], axis =1)
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

X_train = X_train[X_train.Class == 0]

X_train = X_train.drop(['Class'], axis=1)

y_test = X_test['Class']

X_test = X_test.drop(['Class'], axis=1)

X_train = X_train

X_test = X_test

print(X_train.shape)



print(X_test.shape)

print(y_test.shape)
scaler = StandardScaler().fit(X_train.Amount.values.reshape(-1,1))

X_train['Amount'] = scaler.transform(X_train.Amount.values.reshape(-1,1))

X_test['Amount'] = scaler.transform(X_test.Amount.values.reshape(-1,1))
X_train.shape
from keras.layers import Input, Dense

from keras import regularizers

from keras.models import Model, load_model

from keras.callbacks import ModelCheckpoint, TensorBoard
input_dim = X_train.shape[1]

encoding_dim = 14

input_layer = Input(shape=(input_dim,))

encoder = Dense(encoding_dim, activation="tanh", 

                activity_regularizer=regularizers.l1(10e-5))(input_layer)

encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)



decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)



autoencoder = Model(inputs=input_layer, outputs=decoder)
X_train.shape
nb_epoch = 100

batch_size = 32



autoencoder.compile(optimizer='adam', 

                    loss='mean_squared_error', 

                    )



checkpointer = ModelCheckpoint(filepath="model.h5",

                               verbose=0,

                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',

                          histogram_freq=0,

                          write_graph=True,

                          write_images=True)



history = autoencoder.fit(X_train, X_train,

                    epochs=nb_epoch,

                    batch_size=batch_size,

                    shuffle=True,

                    validation_split=0.3,

                    verbose=1,

                    callbacks=[checkpointer, tensorboard]).history
plt.figure(figsize = (10,5))

plt.plot(history['loss'], label = 'Training Loss')

plt.plot(history['val_loss'], label = 'CV Loss')

plt.title("Model Loss")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.grid()

plt.legend()

plt.show()
autoencoder = load_model('model.h5')
predictions = autoencoder.predict(X_test)
predictions.shape
mse = np.mean(np.power(X_test - predictions, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class':y_test})
error_df.groupby(['true_class']).describe()
plt.figure(figsize = (10,5))

sns.distplot(error_df[error_df['true_class']==0]['reconstruction_error'], bins = 5, label = 'Normal')

sns.distplot(error_df[error_df['true_class']==1]['reconstruction_error'], bins=5, label = 'Fraud')

plt.legend()

plt.show()

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support, roc_auc_score)
threshold = 1.4
groups = error_df.groupby('true_class')

fig, ax = plt.subplots(figsize = (20,8))



for name, group in groups:

    ax.plot(group.index, group.reconstruction_error, marker='+', ms=10, linestyle='',

            label= "Fraud" if name == 1 else "Normal")

ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')

ax.legend()

plt.title("Reconstruction error for different classes")

plt.ylabel("Reconstruction error")

plt.xlabel("Data point index")

plt.grid()

plt.show();

fpr, tpr, thres = roc_curve(error_df.true_class, error_df.reconstruction_error)

plt.plot(fpr, tpr, label = 'AUC') 

plt.plot([0,1], [0,1], ':', label = 'Random') 

plt.legend() 

plt.grid() 

plt.ylabel("TPR") 

plt.xlabel("FPR") 

plt.title('ROC') 

plt.show() 
LABELS = ['Normal', 'Fraud']

threshold = 2

y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.true_class, y_pred)

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap='Greens');

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()

## Chosen metric is AUC ROC as data is imbalanced

print("Area under ROC : ", roc_auc_score(error_df.true_class,y_pred ))
print(classification_report(error_df.true_class,y_pred))