!pip install tensorflow-gpu==2.0.0-alpha
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

import tensorflow as tf

import seaborn as sns

import pandas as pd

import numpy as np 

import warnings

import os



print(os.listdir("../input"))
warnings.filterwarnings('ignore')
data = pd.read_csv("../input/framingham.csv")
data.head()
data.shape
columns = data.columns
for col in columns:

    print(col, sum(pd.isnull(data[col])))
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')



imp.fit(data)



new_data = imp.transform(data)
data = pd.DataFrame(new_data, columns=columns)
data.head()
for col in columns:

    print(col, sum(pd.isnull(data[col])))
data["TenYearCHD"] = data["TenYearCHD"].map(lambda x: int(x))
sns.countplot(data["TenYearCHD"])

plt.title("Labels")

plt.show()
data.hist(figsize=(20, 20))

plt.show()
labels = data.pop("TenYearCHD").values

data = data.values
x_train, x_test, y_train, y_test = train_test_split(data ,labels,

                                                    test_size=.1,

                                                    random_state=5)
print("shape x_train:", x_train.shape)

print("shape y_train:", y_train.shape)

print("shape x_test:", x_test.shape)

print("shape y_test:", y_test.shape)
def min_max_normalized(data):

    col_max = np.max(data, axis=0)

    col_min = np.min(data, axis=0)

    return np.divide(data - col_min, col_max - col_min)
x_train = min_max_normalized(x_train)

x_test = min_max_normalized(x_test)
x_train = tf.cast(x_train, dtype=tf.float32)

y_train = tf.cast(y_train, dtype=tf.float32)



train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)





x_test = tf.cast(x_test, dtype=tf.float32)

y_test = tf.cast(y_test, dtype=tf.float32)



test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(16, activation='relu', input_shape=(15,)),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(2, activation='softmax')

])
model.compile(optimizer=optimizer,

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
H = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
plt.plot(H.history['accuracy'])

plt.plot(H.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'])

plt.show()



plt.plot(H.history['loss'])

plt.plot(H.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'])

plt.show()