import pandas as pd
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BASE_PATH = '../input/digit-recognizer'

os.listdir(BASE_PATH)
train = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
test = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))

train_x = train.loc[:, train.columns != 'label']
train_y = train['label']
train_x.shape
train_x.head()
train_y.value_counts()
train_x = train_x/255
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
predictions = model(train_x).numpy()
predictions
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(train['label'], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(train.loc[:, train.columns != 'label'], train['label'], epochs=50)
submission = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))
submission.head()
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probabilities = probability_model.predict(test)

results = pd.DataFrame({
    "ImageId": np.arange(1,28001),
    "Label": np.apply_along_axis(lambda x: np.argmax(x), 1, probabilities)    
}, columns=["ImageId", "Label"], index=np.arange(0,28000))

results.head()
results.to_csv('submission.csv',index=False)
