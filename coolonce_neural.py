# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.layers import Flatten, Reshape, GlobalAveragePooling1D

from keras.layers.embeddings import Embedding

from keras.layers import Conv1D, GlobalMaxPooling1D

from sklearn.model_selection import train_test_split

from keras.layers.convolutional import Conv1D, Conv2D

from keras.layers.convolutional import MaxPooling1D,MaxPooling2D

import keras
# nRowsRead = 1000 # specify 'None' if want to read whole file

# Control.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/quality-diplom/Control.csv', delimiter=',')

df1.dataframeName = 'Control.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')

df2 = pd.read_csv('/kaggle/input/quality-diplom/Quality.csv', delimiter='\t')

df2.dataframeName = 'Quality.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df_control = df1

df_quality = df2
df_control = df_control.drop('Unnamed: 0', axis=1).set_index('date')
df_quality = df_quality.drop('Unnamed: 0', axis=1).set_index('date')
df_control.drop([col for col in df1.columns if col.startswith('Wickler')],

                 axis=1, inplace=True)
sum(df_quality['Stippe_-3000'].isna())
df_quality['Stippe_-3000'].median()
df_quality['Stippe_-3000'] = df_quality['Stippe_-3000'].fillna(df_quality['Stippe_-3000'].median())
sum(df_quality['Stippe_-3000'].isna())
df_quality[df_quality['Stippe_-3000'].isna()]['Stippe_-3000']
treshold = 47.5

# y = df_quality['Stippe_-3000'] > treshold

y = df_quality['Stippe_-3000'] > treshold

X = df_control

#ФИНТ УШАМИ)))))

# X['y'] = y

# X = X.T.shape
df_quality['Stippe_-3000']
y = pd.DataFrame(y)

y.head()
y['Stippe_-3000'] = y['Stippe_-3000'].astype(float)
X.head()
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sc = StandardScaler()

X = sc.fit_transform(X)

# y = sc.fit_transform(y)
y = y.astype(float)
X_df_scaller = pd.DataFrame(X, columns=df_control.columns, index=df_control.index)

X_df_scaller.tail()
X_df_scaller = X_df_scaller.merge(y, left_index=True, right_index=True, how='outer')
X_df_scaller.head()
X = X_df_scaller.values
def split_sequences(sequences, n_steps):

	X, y = list(), list()

	for i in range(len(sequences)):

		# find the end of this pattern

		end_ix = i + n_steps

		# check if we are beyond the dataset

		if end_ix > len(sequences):

			break

		# gather input and output parts of the pattern

		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]

		X.append(seq_x)

		y.append(seq_y)

	return np.array(X), np.array(y)
n_steps = 5
X, y = split_sequences(X_df_scaller.values, n_steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape
n_features = X.shape[2]
(n_steps, n_features)
from keras.utils.vis_utils import plot_model
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[0]

verbose, epochs, batch_size, n_filters = 0, 2, 32, 32

n_kernel = 2

model = Sequential()



model.add(Conv1D(filters=256, kernel_size=1, dilation_rate=1, padding='valid', activation='relu', input_shape=(n_timesteps,n_features)))

model.add(Conv1D(filters=128, kernel_size=1, dilation_rate=1, padding='valid', activation='relu'))

model.add(MaxPooling1D(pool_size=4))

model.add(MaxPooling1D(pool_size=1))

model.add(MaxPooling1D(pool_size=1))

model.add(Flatten())

model.add(Dense(50, activation='relu'))

model.add(Dense(1))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# plot_model(model_cnn_02, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#binary_crossentropy
callbacks_list = [

#     keras.callbacks.ModelCheckpoint(

#         filepath='best_model.{epoch:02d}-val_accuracy:{val_accuracy:.2f}.h5',

#         monitor='val_accuracy', save_best_only=True),

    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15)

]



history = model.fit(X_train, y_train, epochs=10, batch_size=batch_size,validation_split=0.3, verbose=1,callbacks=callbacks_list,)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label="loss")

plt.plot(history.history['val_loss'], label="val_loss")

plt.plot(history.history['accuracy'], label="accuracy")





plt.plot(history.history['val_accuracy'], label="val_accuracy")

plt.xlabel('epoch')

plt.legend()

plt.show()

yhat_02 = model.predict(X_test)

predicts02 = yhat_02[:,0]

roc_auc_score(y_test, predicts02)

# yhat_02
fpr, tpr, threshold = roc_curve(y_test, predicts02)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,7))

plt.plot(fpr, tpr, label='cnn (area = %0.3f)' % roc_auc, linewidth=2)

# plt.plot(fpr_cnn, tpr_cnn, label='w2v-CNN (area = %0.3f)' % roc_auc_nn, linewidth=2)



plt.plot([0, 1], [0, 1], 'k--', linewidth=2)

plt.xlim([-0.05, 1.0])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive Rate', fontsize=18)

plt.ylabel('True Positive Rate', fontsize=18)

plt.title('Receiver operating characteristic: is positive', fontsize=18)

plt.legend(loc="lower right")

plt.show()
model.save_weights("model_roc="+str(round(roc_auc,2))+".h5")

# predicts = yhat[:,0]

predicts02 = yhat_02[:,0]
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc

# predicts = predicts  > 0.99

predicts02 = predicts02 > 0.99
# predicts = predicts.astype(int)

predicts02 = predicts02.astype(int)
roc_auc_score(y_test, predicts02)
confusion_matrix(y_test, predicts02)
fpr, tpr, threshold = roc_curve(y_test, predicts02)

roc_auc = auc(fpr, tpr)
# fpr_cnn, tpr_cnn, threshold = roc_curve(y_test, predicts)

# roc_auc_nn = auc(fpr_cnn, tpr_cnn)

plt.figure(figsize=(8,7))

plt.plot(fpr, tpr, label='cnn (area = %0.3f)' % roc_auc, linewidth=2)

# plt.plot(fpr_cnn, tpr_cnn, label='w2v-CNN (area = %0.3f)' % roc_auc_nn, linewidth=2)



plt.plot([0, 1], [0, 1], 'k--', linewidth=2)

plt.xlim([-0.05, 1.0])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive Rate', fontsize=18)

plt.ylabel('True Positive Rate', fontsize=18)

plt.title('Receiver operating characteristic: is positive', fontsize=18)

plt.legend(loc="lower right")

plt.show()