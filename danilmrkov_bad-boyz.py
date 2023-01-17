# TF libraries
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import efficientnet.keras as efn
from keras.utils import plot_model

# ML libraries
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib.image import imread
import matplotlib.pyplot as plt
%matplotlib inline
font = {
    'family': 'serif',
    'color':  'darkred',
    'weight': 'bold',
    'size': 22,
}
SEED = 257

TRAIN_DIR = 'train/'
TEST_DIR = 'test/'
categories = ['hot dog', 'not hot dog']
X, y = [], []

for category in categories:
    category_dir = os.path.join(TRAIN_DIR, category)
    for image_path in os.listdir(category_dir):
        X.append(imread(os.path.join(category_dir, image_path)))
        y.append(category)
y = [1 if x == 'hot dog' else 0 for x in y]
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.25, random_state=SEED)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# CNN architecture
efficient_net = efn.EfficientNetB3(
    weights='imagenet',
    input_shape=(100,100,3),
    include_top=False,
    pooling='max'
)

HOTDOG_DESTROYER = Sequential()
HOTDOG_DESTROYER.add(efficient_net)

HOTDOG_DESTROYER.add(Dense(128, activation='relu'))
HOTDOG_DESTROYER.add(BatchNormalization())
HOTDOG_DESTROYER.add(Dropout(0.3))

HOTDOG_DESTROYER.add(Dense(128, activation='relu'))
HOTDOG_DESTROYER.add(BatchNormalization())
HOTDOG_DESTROYER.add(Dropout(0.3))

HOTDOG_DESTROYER.add(Dense(64, activation='relu'))
HOTDOG_DESTROYER.add(BatchNormalization())

HOTDOG_DESTROYER.add(Dense(units = 1, activation='sigmoid'))
# Compiler and summary
HOTDOG_DESTROYER.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

HOTDOG_DESTROYER.summary()
# Model fit
history = HOTDOG_DESTROYER.fit(
    X_train, y_train,
    batch_size=32,
    epochs=150,
    verbose=1,
    validation_data=(X_test, y_test)
) 
# Evaluation
score = HOTDOG_DESTROYER.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# ROC-AUC score
roc_auc_score(y_test, HOTDOG_DESTROYER.predict_proba(X_test))
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('Accuracy')
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
leaderboard_X = []  
leaderboard_filenames = []
for image_path in os.listdir(TEST_DIR):
    leaderboard_X.append(imread(os.path.join(TEST_DIR, image_path)))
    leaderboard_filenames.append(image_path)
plt.axis("off");
plt.imshow(leaderboard_X[0]);
print(leaderboard_X[0].shape, leaderboard_filenames[0])
leaderboard_X = np.array(leaderboard_X)
pred = HOTDOG_DESTROYER.predict(leaderboard_X)
pred.tolist()
leadeboard_predictions = []

for i in range(len(pred)):
    leadeboard_predictions.append(float(pred[i]))
  
leadeboard_predictions
pred.round(8)
idx = 519

plt.axis("off");
if leadeboard_predictions[idx] > 0.5:
    plt.text(20, -5, 'HOT DOG!!!', fontdict=font)
else:
    plt.text(15, -5,'not hot dog...', fontdict=font)
plt.imshow(leaderboard_X[idx]);
pd.set_option('display.float_format', lambda x: '%.10f' % x)
submission = pd.DataFrame(
    {
        'image_id': leaderboard_filenames, 
        'image_hot_dog_probability': leadeboard_predictions
    }
)
submission.head()
submission.to_csv('HOTDOG_DESTROYER.csv', index=False, float_format='%.10f')