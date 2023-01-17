import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
test = pd.read_csv('../input/digit-recognizer/test.csv')
train = pd.read_csv('../input/digit-recognizer/train.csv')
digit_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train_label = train['label']
train_image = train.drop('label', axis=1)
train_image_array = train_image.values
train_image_array = train_image_array.astype(np.float)
test_array = test.values
test_array = test_array.astype(np.float)
train_image_array, test_array = train_image_array / 255.0, test_array / 255.0
def display(img):
  image = img.reshape(28, 28)

  plt.axis('off')
  plt.imshow(image)
  

display(train_image_array[7])
train_image_3d = []
for i in range(len(train_image_array)):
  one_img = train_image_array[i].reshape(28, 28)
  train_image_3d.append(one_img)

train_image_4d = np.array(train_image_3d)[:, :, :, np.newaxis]
test_3d = []
for i in range(len(test_array)):
  one_img = test_array[i].reshape(28, 28)
  test_3d.append(one_img)

test_4d = np.array(test_3d)[:, :, :, np.newaxis]
model = tf.keras.models.Sequential([
                                    Conv2D(64, 3, activation='relu'),
                                    MaxPooling2D(2, 2),
                                    Conv2D(32, 3, activation='relu'),
                                    MaxPooling2D(2, 2),
                                    Flatten(),
                                    Dense(128, activation='relu'),
                                    Dense(10, activation='softmax')
])

model.compile(optimizer='Adagrad',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_index, valid_index in cv.split(train_image_4d, train_label):
  model.fit(train_image_4d[train_index], train_label[train_index], epochs=10)
  scores = model.evaluate(train_image_4d[valid_index], train_label[valid_index], verbose=2)
predictions = model.predict(test_4d)
predictions_list = [np.argmax(x) for x in predictions]
digit_submission['Label'] = predictions_list
digit_submission