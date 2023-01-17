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
import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')



seed = 42

train_df, test_df = train_test_split(df, test_size=0.1, random_state=seed)

train_df, val_df = train_test_split(train_df, test_size=1/9, random_state=seed)
df.Class.value_counts()
train_df.Class.value_counts()
val_df.Class.value_counts()
test_df.Class.value_counts()
models = {}

history = {}

test_summary = []

prediction_probs = []

model_names = []
train_df_copy = train_df.copy()

val_df_copy = val_df.copy()

test_df_copy = test_df.copy()
X_train, y_train = train_df_copy.drop('Class', axis=1).values, train_df_copy.Class.values

X_val, y_val = val_df_copy.drop('Class', axis=1).values, val_df_copy.Class.values

X_test, y_test = test_df_copy.drop('Class', axis=1).values, test_df_copy.Class.values



min_max_scaler = MinMaxScaler(feature_range=(0, 1))

X_train = min_max_scaler.fit_transform(X_train)

X_val = min_max_scaler.transform(X_val)

X_test = min_max_scaler.transform(X_test)
model_1 = tf.keras.Sequential([

    tf.keras.layers.Dense(50, input_shape=[X_train.shape[1]], activation='relu'),

#     tf.keras.layers.Dense(50, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

model_1.summary()
history['model_1'] = model_1.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=200)

models['model_1'] = model_1
his = history['model_1']
plt.figure(figsize=(12, 8))

plt.plot(his.history['loss'], label='Train loss')

plt.plot(his.history['val_loss'], label='Val loss')

plt.legend()

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.savefig('model_1_fig.png');
y_prob = model_1.predict(X_test)

y_pred = (y_prob > 0.5).astype(np.int)

confusion_matrix(y_test, y_pred)
name = 'Shallow ANN on full dataset'

record = {'Name': name,

          'Precision': precision_score(y_test, y_pred),

          'Recall': recall_score(y_test, y_pred),

          'F1-score': f1_score(y_test, y_pred),

          'AUC': roc_auc_score(y_test, y_prob)}

test_summary.append(record)

model_names.append(name)

prediction_probs.append(y_prob)
# Change threshold

y_pred = model_1.predict(X_test)

y_pred = (y_pred > 0.05).astype(np.int)

cm = confusion_matrix(y_test, y_pred)

print(cm)
model_2 = tf.keras.Sequential([

    tf.keras.layers.Dense(50, input_shape=[X_train.shape[1]], activation='relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(50, activation='relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(50, activation='relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

model_2.summary()
history['model_2'] = model_2.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=200)

models['model_2'] = model_2
his = history['model_2']

plt.figure(figsize=(12, 8))

plt.plot(his.history['loss'], label='Train loss')

plt.plot(his.history['val_loss'], label='Val loss')

plt.legend()

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.savefig('model_2_fig.png');
y_prob = model_2.predict(X_test)

y_pred = (y_prob > 0.5).astype(np.int)

confusion_matrix(y_test, y_pred)
name = 'Deep ANN on full dataset'

record = {'Name': name,

          'Precision': precision_score(y_test, y_pred),

          'Recall': recall_score(y_test, y_pred),

          'F1-score': f1_score(y_test, y_pred),

          'AUC': roc_auc_score(y_test, y_prob)}

test_summary.append(record)

model_names.append(name)

prediction_probs.append(y_prob)
train_df_copy = train_df.copy()

val_df_copy = val_df.copy()

test_df_copy = test_df.copy()
from imblearn.over_sampling import SMOTE
X_train, y_train = train_df_copy.drop('Class', axis=1).values, train_df_copy.Class.values

X_val, y_val = val_df_copy.drop('Class', axis=1).values, val_df_copy.Class.values

X_test, y_test = test_df_copy.drop('Class', axis=1).values, test_df_copy.Class.values



seed = 42

X_train, y_train = SMOTE(random_state=seed).fit_sample(X_train, y_train)



min_max_scaler = MinMaxScaler(feature_range=(0, 1))

X_train = min_max_scaler.fit_transform(X_train)

X_val = min_max_scaler.transform(X_val)

X_test = min_max_scaler.transform(X_test)
model_3 = tf.keras.Sequential([

    tf.keras.layers.Dense(50, input_shape=[X_train.shape[1]], activation='relu'),

#     tf.keras.layers.Dense(30, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

model_3.summary()
history['model_3'] = model_3.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=200)

models['model_3'] = model_3
his = history['model_3']

plt.figure(figsize=(12, 8))

plt.plot(his.history['loss'], label='Train loss')

plt.plot(his.history['val_loss'], label='Val loss')

plt.legend()

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.savefig('model_3_fig.png');
y_prob = model_3.predict(X_test)

y_pred = (y_prob > 0.5).astype(np.int)

confusion_matrix(y_test, y_pred)
name = 'Shallow ANN on oversampled dataset'

record = {'Name': name,

          'Precision': precision_score(y_test, y_pred),

          'Recall': recall_score(y_test, y_pred),

          'F1-score': f1_score(y_test, y_pred),

          'AUC': roc_auc_score(y_test, y_prob)}

test_summary.append(record)

model_names.append(name)

prediction_probs.append(y_prob)
model_4 = tf.keras.Sequential([

    tf.keras.layers.Dense(50, input_shape=[X_train.shape[1]], activation='relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(50, activation='relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(50, activation='relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model_4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

model_4.summary()
history['model_4'] = model_4.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=200)

models['model_4'] = model_4
his = history['model_4']

plt.figure(figsize=(12, 8))

plt.plot(his.history['loss'], label='Train loss')

plt.plot(his.history['val_loss'], label='Val loss')

plt.legend()

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.savefig('model_4_fig.png');
y_prob = model_4.predict(X_test)

y_pred = (y_prob > 0.5).astype(np.int)

confusion_matrix(y_test, y_pred)
name = 'Deep ANN on oversampled dataset'

record = {'Name': name,

          'Precision': precision_score(y_test, y_pred),

          'Recall': recall_score(y_test, y_pred),

          'F1-score': f1_score(y_test, y_pred),

          'AUC': roc_auc_score(y_test, y_prob)}

test_summary.append(record)

model_names.append(name)

prediction_probs.append(y_prob)
fpr = dict()

tpr = dict()

auc = dict()

for i, name in enumerate(model_names):

    fpr[name], tpr[name], _ = roc_curve(y_test, prediction_probs[i])

    auc[name] = roc_auc_score(y_test, prediction_probs[i])
plt.figure(figsize=(12, 8))

lw = 2

for name in fpr.keys():

    plt.plot(fpr[name], tpr[name], lw=lw, label='ROC curve of {} (area = {:.4f})'.format(name, auc[name]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('ROC_ann.png');
test_summary = pd.DataFrame(test_summary)

test_summary
test_summary.to_csv('NN_summary.csv')