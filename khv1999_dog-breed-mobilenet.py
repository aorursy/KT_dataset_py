# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import tensorflow_hub as hub

print(tf.__version__)
print(hub.__version__)
labels = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
labels.head()
train_path = '/kaggle/input/dog-breed-identification/train/'
filenames = [train_path + fname + ".jpg" for fname in labels['id']]
len(filenames)
import os
len(os.listdir(train_path))
from IPython.display import display, Image
Image(filenames[420])
Image(filenames[69])
labels = labels['breed'].to_numpy()
labels[:5]
unique_breed = np.unique(labels)
boolean_labels = [label==unique_breed for label in labels]
boolean_labels[:2]
X = filenames
y = boolean_labels
NUM_IMAGES = 1000
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES], y[:NUM_IMAGES], test_size=0.2)
len(X_train), len(y_train), len(X_val), len(y_val)
IMG_SIZE = 224

def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image
#to return image tuple and use that to create batches of data 
def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label
BATCH_SIZE = 32

def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    #if test data we dont have y(labels)
    if test_data:
        print("Creating test data batches")
        #this basically converts the x and y that we input into tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
        #in this we map it to preprocessing function that we wrote and create batches
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch
    elif valid_data:
        #no need to shuffle
        print("Creating validation data batches")
        
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch
    else:
        #train data so shuffle
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        data = data.shuffle(buffer_size = len(x))
        data = data.map(get_image_label)
        data_batch = data.batch(BATCH_SIZE)
        
    return data_batch
        
        
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)
train_data.element_spec, val_data.element_spec
import matplotlib.pyplot as plt
def show_25_images(images, labels):
    plt.figure(figsize=(10, 10))
    
    for i in range(25):
        ax = plt.subplot(5, 5, i+1)
        plt.imshow(images[i])
        plt.title(unique_breed[labels[i].argmax()])
        plt.axis('off')
train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images, train_labels)
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]
OUTPUT_SHAPE = len(unique_breed)

# Setup model URL from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
def create_model(input_shape = INPUT_SHAPE, output_shape = OUTPUT_SHAPE, model_url=MODEL_URL):
    
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url),
        tf.keras.layers.Dense(units = output_shape, activation='softmax')
    ])
    
    model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(),
        metrics = ['accuracy']
    )
    
    model.build(INPUT_SHAPE)
    return model
model = create_model()
model.summary()
%load_ext tensorboard
import datetime

def create_tensorboard_callback():
    logdir = os.path.join("logs",
                         datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(logdir)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',
                                                  patience=3)
NUM_EPOCHS = 100
def train_model():
    model = create_model()
    tensorboard = create_tensorboard_callback()
    
    model.fit(x = train_data,
              epochs = NUM_EPOCHS,
              validation_data = val_data,
              validation_freq=1,
              callbacks=[tensorboard, early_stopping])
    return model
model = train_model()
%tensorboard --logdir logs
predictions = model.predict(val_data, verbose=True)
def get_pred_label(prediction_probabilities):
    return unique_breed[np.argmax(prediction_probabilities)]
get_pred_label(predictions[0])
def unbatchify(data):
    images = []
    labels = []
    
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(unique_breed[np.argmax(label)])
    return image, labels
val_images, val_labels = unbatchify(val_data)
val_images[0], val_labels[0]
full_data = create_data_batches(X, y)
full_model = create_model()
full_model_tensorboard = create_tensorboard_callback()
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy',
                                                             patience = 3)
full_model.fit(x = full_data,
               epochs = NUM_EPOCHS,
               callbacks = [full_model_tensorboard,
                            full_model_early_stopping])
preds_df = pd.DataFrame(columns=['id'] + list(unique_breed))
preds_df.head()
test_path = '/kaggle/input/dog-breed-identification/test/'
preds_df['id'] = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df.head()
test_path = "../input/dog-breed-identification/test/"
test_filenames = [test_path + fname for fname in os.listdir(test_path)]

test_filenames[:10]
test_data = create_data_batches(test_filenames, test_data=True)
test_prediction = full_model.predict(test_data, verbose=1)
preds_df[list(unique_breed)] = test_prediction
preds_df.head()
preds_df.to_csv('submission.csv', index=False)
