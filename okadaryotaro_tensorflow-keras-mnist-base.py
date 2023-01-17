import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#%tensorflow_version 2.x # Google Colaboratoryでは必要だがいらない

import tensorflow as tf
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

x_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
x_train = np.array(train_df.drop(columns='label'))

y_train = np.array(train_df['label'])

x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape)
model = tf.keras.models.Sequential([

  #tf.keras.layers.Flatten(input_shape=(28,28)), # 入力データがすでにフラットなので要らない

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10, activation='softmax'),

])

model.compile(optimizer='adam',

  loss='sparse_categorical_crossentropy',

  metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
result = model.predict(x_test)
print(result[:5]) # 結果のデータの確認

print([x.argmax() for x in result[:5]]) 

y_test = [x.argmax() for x in result]
submit_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submit_df.head()
submit_df['Label'] = y_test

submit_df.head()
submit_df.to_csv('submission.csv', index=False)