import tensorflow as tf

#tf.debugging.set_log_device_placement(True)

print(tf.__version__)



print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

device_name = tf.test.gpu_device_name()

if "GPU" not in device_name:

    print("GPU device not found")

print('Found GPU at: {}'.format(device_name))



from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input,Dense, Dropout
import matplotlib.pyplot as plt

import seaborn as sns
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
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

pd.options.display.float_format = "{:.4f}".format
print("shape:", df.shape)
df.columns
df.info()
df.describe()
df.head(10)
# Sample figsize in inches

fig, ax = plt.subplots(figsize=(20,10))         

# Imbalanced DataFrame Correlation

corr = df.corr()

sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)

ax.set_title("Imbalanced Correlation Matrix", fontsize=14)

plt.show()
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='minority', random_state=7)
resampled_X, resampled_Y = sm.fit_resample(df.drop('Class', axis=1), df['Class'])

oversampled_df = pd.concat([pd.DataFrame(resampled_X), pd.DataFrame(resampled_Y)], axis=1)

oversampled_df.columns = df.columns

oversampled_df['Class'].value_counts()
# Sample figsize in inches

fig, ax = plt.subplots(figsize=(20,10))         

# Imbalanced DataFrame Correlation

corr = oversampled_df.corr()

sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)

ax.set_title("Imbalanced Correlation Matrix", fontsize=14)

plt.show()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = oversampled_df.iloc[:, 1:-1].values

y = oversampled_df.iloc[:, -1].values

y = y.reshape(-1, 1)

print(X.shape, y.shape)



X = sc.fit_transform(X)

print(X[0])
x_features = X.shape[1]

y_features = y.shape[1]

print("x_features: ", x_features)

print("y_features: ", y_features)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("train shapes:", x_train.shape, y_train.shape)

print("test shapes:", x_test.shape, y_test.shape)
adam = tf.keras.optimizers.Adam(learning_rate=0.0005)



i = Input(shape=(x_features,))

x = Dense(64, activation="relu")(i)

x = Dense(64, activation="relu")(x)

o = Dense(y_features, activation="sigmoid")(x)



model = Model(i,o)

model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=adam)

model.summary()
callback = tf.keras.callbacks.EarlyStopping(

    monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',

    baseline=None, restore_best_weights=True

)

r = model.fit(x_train, y_train, epochs=100, batch_size=2048, verbose=1, validation_data=(x_test, y_test), callbacks=[callback])
results = model.evaluate(x_test, y_test, batch_size=5, verbose=1)

print("Loss: %.2f" % results[0])

print("Acc: %.2f" % results[1])
print(r.history.keys())

plt.plot(r.history['loss'])

plt.plot(r.history['val_loss'])

plt.legend(['loss', 'val_loss'])

plt.show()
plt.plot(r.history['accuracy'])

plt.plot(r.history['val_accuracy'])

plt.legend(['accuracy', 'val_accuracy'])

plt.show()
y_pred = model.predict(x_test)

y_pred = np.round(y_pred, decimals=0).astype(int)

#y_pred = np.argmax(y_pred,axis=-1)

#y_pred = y_pred.astype(int)

y_pred
df_pred = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1)

df_pred.columns = df.drop('Time', axis=1).columns

df_pred.rename(columns={"Class":"Old_class"}, inplace=True)

df_pred['New_class'] = y_pred

df_pred.head()
cm = pd.crosstab(df_pred["New_class"], df_pred['Old_class'])

cm
true_pos = np.sum(np.diag(cm))

false_pos = cm[0][1]

false_neg = cm[1][0]

#tot = np.sum(np.sum(cm, axis=0))

precision = true_pos / (true_pos + false_pos) * 100

recall = true_pos / (true_pos + false_neg) * 100

f1 = 2 * (precision * recall) / (precision + recall)

print("Precision: %.3f%%" % (precision))

print("Recall: %.3f%%" % (recall))

print("F1: %.3f%%" % (f1))