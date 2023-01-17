import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from kaggle_datasets import KaggleDatasets
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [16, 8]

print('Using Tensorflow version:', tf.__version__)
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path('shopee-product-detection-open')

# Configuration
EPOCHS = 110
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
image_size=(299, 299)
train_df = pd.read_csv('/kaggle/input/shopee-product-detection-open/train.csv')
test_df = pd.read_csv('/kaggle/input/shopee-product-detection-open/test.csv')

train_df.shape, test_df.shape
train_df.head()
def show_train_img(category):
    
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))
    
    train_path = '/kaggle/input/shopee-product-detection-open/train/train/train/'
    ten_random_samples = pd.Series(os.listdir(os.path.join(train_path, category))).sample(10).values
    
    for idx, image in enumerate(ten_random_samples):
        final_path = os.path.join(train_path, category, image)
        img = cv2.imread(final_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes.ravel()[idx].imshow(img)
        axes.ravel()[idx].axis('off')
    plt.tight_layout()
show_train_img('38')
show_train_img('12')
show_train_img('32')
def show_test_img():
    
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 10))
    
    test_path = '/kaggle/input/shopee-product-detection-open/test/test/test/'
    ten_random_samples = pd.Series(os.listdir(test_path)).sample(10).values
    
    for idx, image in enumerate(ten_random_samples):
        final_path = os.path.join(test_path, image)
        img = cv2.imread(final_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes.ravel()[idx].imshow(img)
        axes.ravel()[idx].axis('off')
    plt.tight_layout()
show_test_img()
# pick random samples

dataset_path = {}

categories = np.sort(train_df['category'].unique())

for cat in categories:
#     try:
#         dataset_path[cat] = train_df[train_df['category'] == cat]['filename'].sample(2100)
#     except:
    dataset_path[cat] = train_df[train_df['category'] == cat]['filename'].sample(frac=1.)
category_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
                 '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                 '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                 '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                 '40', '41']

class_weight = {v:1.0 for v in range(42)}
class_weight[0]=2.0
class_weight[1]=2.0
class_weight[2]=2.0
class_weight[3]=2.0
class_weight[20]=2.0
class_weight[36]=2.0
class_weight[41]=2.0
train_paths = []

for idx, key in enumerate(dataset_path.keys()):
    if key == idx:
        for path in dataset_path[idx]:
            train_paths.append(os.path.join(GCS_DS_PATH, 'train', 'train', 'train', category_list[idx], path))
labels = []

for label in dataset_path.keys():
    labels.extend([label] * len(dataset_path[label]))
from tensorflow.keras.utils import to_categorical

# convert to numpy array
train_paths = np.array(train_paths)

# convert to one-hot-encoding-labels
train_labels = to_categorical(labels)
from sklearn.model_selection import train_test_split

train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, 
                                                                        train_labels, 
                                                                        stratify=train_labels,
                                                                        test_size=0.1, 
                                                                        random_state=2020)

train_paths.shape, valid_paths.shape, train_labels.shape, valid_labels.shape
test_paths = []

for path in test_df['filename']:
    test_paths.append(os.path.join(GCS_DS_PATH,  'test', 'test', 'test', path))
    
test_paths = np.array(test_paths)
def decode_image(filename, label=None, image_size=image_size):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label
    
def decode_aug_image(filename, label=None, image_size=image_size):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    #augment
    image = tf.image.random_brightness(image, 0.4)
    image = tf.image.random_contrast(image, 0.2, 0.5)
    image = tf.image.random_crop(image, [200, 200, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label
    

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_aug_image, num_parallel_calls=AUTO)
    .cache()
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)
!pip install -q efficientnet
from tensorflow.keras.layers import Dense, Dropout
import efficientnet.tfkeras as efn 
from functools import partialmethod
import transformers
from transformers.optimization_tf import AdamWeightDecay, create_optimizer
initial_lr = 1e-4
LABEL_SMOOTHING = 0.10
def loss_with_ls(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, \
                                                    label_smoothing=LABEL_SMOOTHING)
def build_model(update_model=None):
    with strategy.scope():
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=initial_lr, 
                        decay_steps = 320, end_learning_rate=3e-6, power=1.0,cycle=True)
    
        lr_schedule = transformers.optimization_tf.WarmUp(initial_learning_rate=initial_lr, 
                                                          decay_schedule_fn=lr_schedule, warmup_steps=50)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        if update_model != None:
            update_model.trainable = True
            for layer in update_model.layers:
                layer.trainable = False
                if (layer.name=='stem_conv'):
                    layer.trainable = True
                    break
            update_model.compile(optimizer=optimizer, loss=loss_with_ls, metrics=['accuracy'])
            print("trainable layers",len(update_model.trainable_variables))
            return update_model

        base_model = efn.EfficientNetB5(weights='noisy-student',
                                        include_top=False,
                                        input_shape=image_size+(3,),
                                        pooling='avg')   
        base_model.trainable = False
        x = tf.keras.layers.Dense(42,activation="softmax")(base_model.output)

        model = tf.keras.Model(base_model.input, x)  
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
        print("trainable layers",len(model.trainable_variables))
        return model
model_load = "../input/efficient-netb5/eff_b5.h5"
model_save= "./eff_b5.h5"
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001,
                                   patience=3, verbose=1, mode='auto')    
sv = tf.keras.callbacks.ModelCheckpoint(model_save, \
                                        monitor='val_loss', verbose=1, save_best_only=True,\
                                        save_weights_only=True, mode='auto', save_freq='epoch')


model = build_model()
n_steps = train_labels.shape[0] // BATCH_SIZE

model.fit(
    train_dataset, 
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    class_weight=class_weight,
    epochs=10,
)
# model = build_model(model)
# model.load_weights(model_load)

print("unfreezing few more layers and finetuning")
# unfreeze a few more layers and train till convergence
model = build_model(model)
history = model.fit(
    train_dataset, 
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    class_weight=class_weight,
    initial_epoch=10,callbacks=[early_stopping,sv],
    epochs=EPOCHS,
)

print('Loading best model...')
model.load_weights(model_save)

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
training_loss
test_loss
history.history['val_accuracy']
pred = model.predict(test_dataset, verbose=1)
np.save("pred_prob.npy",pred)
# drop existing feature
test_df = test_df.drop('category', axis=1)

# change with prediction
test_df['category'] = pred.argmax(axis=1)

# then add zero-padding
test_df['category'] = test_df['category'].apply(lambda x: str(x).zfill(2))
test_df.to_csv('submission.csv', index=False)
pred_train = model.predict(test_dataset, verbose=1, )
pred_val = model.predict(test_dataset, verbose=1)
np.save("pred_train_prob.npy",pred_train)
np.save("pred_val_prob.npy",pred_val)
test_df.head()