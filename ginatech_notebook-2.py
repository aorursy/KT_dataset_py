import os

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

import numpy as np

import seaborn as sns



from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, precision_recall_curve
TRAIN = '../input/utensils-traincsv/utensils_train.csv'

TEST = '../input/utensils-testcsv/utensils_test.csv'

CKPT_DIR = 'checkpoints/vgg_16_ckpts_{epoch:03d}.ckpt'

BEST_DIR = 'checkpoints/vgg_16_best.ckpt'
train = pd.read_csv(TRAIN)

train.head()
test = pd.read_csv(TEST)

test.head()
train_y = train['Label'].values

train_y[:5]
test_y = test['Label'].values

test_y[:5]
train_y_encoder = OneHotEncoder(sparse = False)

train_y_encoded = train_y_encoder.fit_transform(train_y.reshape(-1, 1))

train_y_encoded[:5]
test_y_encoder = OneHotEncoder(sparse=False)

test_y_encoded = test_y_encoder.fit_transform(test_y.reshape(-1, 1))

test_y_encoded[:5]
train_X = train.drop('Label', axis=1).values

train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test.drop('Label', axis=1).values

test_X = test_X.reshape(-1, 28, 28, 1)
for i in range(10):

    plt.imshow(train_X[i].reshape(28, 28))

    plt.show()

    print('Label:', train_y[i])
input_ = tf.keras.Input((28, 28, 1))

conv1 = tf.keras.layers.Conv2D(8, (4, 4), activation='relu')(input_)

mp1 = tf.keras.layers.MaxPool2D((2, 2))(conv1)

fl = tf.keras.layers.Flatten()(mp1)

dense1 = tf.keras.layers.Dense(6, activation='relu')(fl)

dense1 = tf.keras.layers.Dense(6, activation='relu')(dense1)

output = tf.keras.layers.Dense(3, activation='softmax')(dense1)



model = tf.keras.Model(inputs = input_, outputs = output)

model.summary()
model.compile('adam', 'categorical_crossentropy')
tf.keras.utils.plot_model(model)
es = tf.keras.callbacks.EarlyStopping(patience=20)

os.makedirs(os.path.dirname(CKPT_DIR), exist_ok=True)

os.makedirs(os.path.dirname(BEST_DIR), exist_ok=True)

mc = tf.keras.callbacks.ModelCheckpoint(CKPT_DIR)

bm = tf.keras.callbacks.ModelCheckpoint(BEST_DIR, save_best_only=True)

hst = model.fit(train_X, train_y_encoded, batch_size = 32, epochs = 10, validation_split=0.2, callbacks = [es, mc, bm])
!ls checkpoints/
best_model = tf.keras.models.load_model(BEST_DIR)
for i in range(10):

    plt.imshow(test_X[i].reshape(28, 28))

    plt.show()

    print('Label:', test_y[i])
test_y_pred = best_model.predict(test_X)
roc_auc_score(test_y_encoded.flatten(), test_y_pred.flatten())
pr, r, thr = precision_recall_curve(test_y_encoded.flatten(), test_y_pred.flatten())
plt.plot(thr, pr[:-1])

plt.plot(thr, r[:-1])

plt.show()
threshold = 0.99
accuracy_score(test_y_encoded.flatten() > threshold, test_y_pred.flatten() > threshold)
print(classification_report(test_y_encoded.flatten() > threshold, test_y_pred.flatten() > threshold))
sns.heatmap(confusion_matrix(test_y_encoded.flatten() > threshold, test_y_pred.flatten() > threshold), annot=True)

plt.ylabel('True')

plt.xlabel('True')

plt.show()