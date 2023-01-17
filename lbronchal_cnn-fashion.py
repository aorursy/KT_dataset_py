import os
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

print(os.listdir("../input"))
dataset_path = "../input"
train_data = pd.read_csv(os.path.join(dataset_path, "fashion-mnist_train.csv"))
test_data = pd.read_csv(os.path.join(dataset_path, "fashion-mnist_test.csv"))
train_data.head()

features = [c for c in train_data.columns if c != 'label']
X = train_data[features]
y = train_data['label']
labels_dict = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 
               4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
labels_counts = y.value_counts()
xlabels = [labels_dict[i] for i in labels_counts.index]
plt.bar(x=xlabels, height=labels_counts.values)
plt.title("Number of cases per label")
plt.xticks(rotation=70)
plt.show()
N_CLASSES = 10
IMG_HEIGHT = 28
IMG_WIDTH = 28
CHANNELS = 1
target_dims = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

plt.title(labels_dict[y[0]])
plt.imshow(X.iloc[0].values.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()
X = X.values.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(1, figsize=(16, 16))
grid = ImageGrid(fig, 111, nrows_ncols=(2, 5), axes_pad=0.5)

for i, img_path in enumerate(X[:10]):
    ax = grid[i]
    img = X[i].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
plt.show()
# To improve reproducibility ( although using a GPU it is not completely possible as you can see
# in https://www.kaggle.com/lbronchal/keras-gpu-cpu-reproducibility-test )

import tensorflow as tf
import random as rn
from keras import backend as K

os.environ['PYTHONHASHSEED'] = '0'

SEED = 123456
np.random.seed(SEED)
rn.seed(SEED)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(SEED)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Flatten, Dropout, Conv2D, Dense, MaxPooling2D, BatchNormalization, LeakyReLU, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
y_onehot = to_categorical(y)
X_test = test_data[features].values.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
y_test = to_categorical(test_data['label'])
X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=SEED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255
batch_size = 512
data_augmentor = ImageDataGenerator(rotation_range=2,
                                    width_shift_range=2,
                                    height_shift_range=2,
                                    fill_mode="nearest")

data_augmentor.fit(X_train)

train_generator = data_augmentor.flow(X_train, y_train, batch_size=batch_size, seed=SEED)
def create_model():
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(4, 4),
                     strides=2,
                     activation='relu',
                     input_shape=target_dims))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu'))
    model.add(Dropout(0.4))   
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(N_CLASSES, activation='softmax'))    
    
    model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])
    
    return model

model = create_model()
model.summary()
EPOCHS = 60
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

checkpoint = ModelCheckpoint("weights-augmentated-best.hdf5", monitor='val_acc', verbose=0, save_best_only=True, mode='max')

history = model.fit_generator(train_generator, 
                              epochs=EPOCHS, 
                              steps_per_epoch=3*X_train.shape[0] // batch_size,
                              validation_steps=X_val.shape[0] // batch_size,
                              validation_data=(X_val, y_val), 
                              verbose=1, 
                              callbacks=[early_stopping, checkpoint])



# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
model.load_weights("weights-augmentated-best.hdf5")
scores = model.evaluate(X_val, y_val, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predicted_classes = model.predict_classes(X_test)

y_true = y_test.argmax(axis=1)
correct = np.where(predicted_classes == y_true)[0]
incorrect = np.where(predicted_classes != y_true)[0]
plt.figure(figsize=(8, 8))
for i, correct in enumerate(incorrect[:9]):
    plt.subplot(430+1+i)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray')
    plt.title("Pred: {} || Class {}".format(labels_dict[predicted_classes[correct]], labels_dict[y_true[correct]]))
    plt.axis('off')
    plt.tight_layout()
plt.show()
from sklearn.metrics import classification_report, confusion_matrix
target_names = ["Class {}: [{}]".format(i, labels_dict[i]) for i in range(N_CLASSES)]
print(classification_report(y_true, predicted_classes, target_names=target_names))
conf_matrix = confusion_matrix(y_true, predicted_classes)
import itertools
plt.figure(figsize=(8, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)

thresh = conf_matrix.max() / 2
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
