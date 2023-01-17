import os

print(os.listdir("../input"))



import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

import seaborn as sns

import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, classification_report, precision_recall_curve



sns.set()
CKPT_DIR = 'checkpoints/vgg_16_ckpts_{epoch:03d}.ckpt'

BEST_DIR = 'checkpoints/vgg_16_best.ckpt'

train_df = pd.read_csv("../input/utensils_train.csv")

test_df = pd.read_csv("../input/utensils_test.csv")



train_df.head()
train_y = train_df['Label'].values

train_y[:5]
train_y_encoder = OneHotEncoder(sparse=False)

train_y_encoded = train_y_encoder.fit_transform(train_y.reshape(-1, 1))

train_y_encoded[:5]
train_X = train_df.drop('Label', axis=1).values

train_X = train_X.reshape(-1, 28, 28, 1)
for i in range(10):

    plt.imshow(train_X[i].reshape(28, 28))

    plt.grid(None)

    plt.show()

    print('Label:', train_y[i])
input_ = tf.keras.Input((28, 28, 1))

conv1 = tf.keras.layers.Conv2D(10, (3, 3), activation='relu')(input_)

conv2 = tf.keras.layers.Conv2D(30, (3, 3), activation='relu')(conv1)

mp1 = tf.keras.layers.AveragePooling2D((2,2))(conv2)

conv3 = tf.keras.layers.Conv2D(50, (3, 3), activation='relu')(mp1)

conv4 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu')(conv3)

conv5 = tf.keras.layers.Conv2D(60, (3, 3), activation='relu')(conv4)

mp2 = tf.keras.layers.AveragePooling2D((2,2))(conv5)

fl = tf.keras.layers.Flatten()(mp2)

dense1 = tf.keras.layers.Dense(40, activation='relu')(fl)

output = tf.keras.layers.Dense(3, activation='softmax')(dense1)



model = tf.keras.Model(inputs=input_, outputs=output)

model.summary()
model.compile('adam', 'categorical_crossentropy')
tf.keras.utils.plot_model(model)
es = tf.keras.callbacks.EarlyStopping(patience=20)

os.makedirs(os.path.dirname(CKPT_DIR), exist_ok=True)

os.makedirs(os.path.dirname(BEST_DIR), exist_ok=True)

mc = tf.keras.callbacks.ModelCheckpoint(CKPT_DIR)

bm = tf.keras.callbacks.ModelCheckpoint(BEST_DIR, save_best_only=True)



hst = model.fit(train_X, train_y_encoded, batch_size=32, epochs=20, validation_split=0.2, callbacks=[es, mc, bm])
!ls checkpoints/
test_y = test_df['Label'].values

test_X = test_df.drop('Label', axis=1).values

test_X = test_X.reshape(-1, 28, 28, 1)



for i in range(10):

    plt.imshow(test_X[i].reshape(28, 28))

    plt.grid(None)

    plt.show()

    print('Label:', test_y[i])
test_y_encoder = OneHotEncoder(sparse=False)

test_y_encoded = test_y_encoder.fit_transform(test_y.reshape(-1, 1))

test_y_encoded[:5]
best_model = tf.keras.models.load_model(BEST_DIR)

test_y_pred = best_model.predict(test_X)
roc_auc_score(test_y_encoded.ravel(), test_y_pred.ravel())
pr, r, thr = precision_recall_curve(test_y_encoded.ravel(), test_y_pred.ravel())



plt.plot(thr, pr[:-1])

plt.plot(thr, r[:-1])

plt.show()
threshold = 0.955
accuracy_score(test_y_encoded.ravel() > threshold, test_y_pred.ravel() > threshold)
print(classification_report(test_y_encoded.ravel() > threshold, test_y_pred.ravel() > threshold))
sns.heatmap(confusion_matrix(test_y_encoded.ravel() > threshold, test_y_pred.ravel() > threshold), annot=True)

plt.ylabel('True')

plt.xlabel('Pred')

plt.show()