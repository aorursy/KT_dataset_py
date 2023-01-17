!pip install -q tensorflow-gpu==2.0.0-beta1 
import os

import sys

import numpy as np 

import pandas as pd 

import cv2

from PIL import Image 

import pathlib

import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET

import tensorflow as tf

import multiprocessing as mp

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



assert sys.version_info >= (3, 5), 'Python ≥3.5 required'

assert tf.__version__ >= '2.0', 'TensorFlow ≥2.0 required' 



RANDOM_SEED = 12345

np.random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)



AUTOTUNE = tf.data.experimental.AUTOTUNE



TEST_SIZE = 0.2

INPUT_IMAGE_SIZE = 224



BATCH_SIZE = 24

EPOCH = 5
print(os.listdir("../input"))
ANNOTATION_DIR = pathlib.Path('../input/annotations/Annotation/')

IMAGES_DIR = pathlib.Path('../input/images/Images/')



BREED_DIR = [path for path in IMAGES_DIR.iterdir()]

BREED_DIR_NAME = [path.name for path in BREED_DIR]



BREED_CODE_TO_NAME = {breed.split('-')[0]: breed.split('-')[1] for breed in BREED_DIR_NAME}

BREED_NAME_TO_CODE = {v: k for k, v in BREED_CODE_TO_NAME.items()}



BREED_LABEL_TO_CODE = {i: code for i, code in enumerate(BREED_CODE_TO_NAME)}

BREED_CODE_TO_LABEL = {v: k for k, v in BREED_LABEL_TO_CODE.items()}



BREED_LABEL_TO_NAME = {i: BREED_CODE_TO_NAME[code] for i, code in BREED_LABEL_TO_CODE.items()}

BREED_NAME_TO_LABEL = {v: k for k, v in BREED_LABEL_TO_NAME.items()}
def path_to_label(path):

    code = path.stem.split('_')[0]

    return BREED_CODE_TO_LABEL[code]





def get_all_file_path(directory, file_pattern=''):

    paths = list(f for f in directory.rglob('**/*{}'.format(file_pattern)) if f.is_file())

    return sorted(paths, key=str) 

    

    

all_image_paths = get_all_file_path(IMAGES_DIR, '.jpg') # PosixPath

all_image_labels = [path_to_label(path) for path in all_image_paths] # [0,1,2,...]



assert len(all_image_paths)==len(all_image_labels), 'Numbers of images and labels not match! {}!={}'.format(len(all_image_paths), len(all_image_labels))



# Write labels to file 

with open('labels.txt', 'w') as f:

    f.write('\n'.join(BREED_NAME_TO_LABEL))
# Crop and save images according to boundings

IMAGES_CROPPED_DIR = pathlib.Path('/tmp/images_cropped/')

IMAGES_CROPPED_DIR.mkdir(parents=True, exist_ok=True) 



# Gets object boundings

def parse_bounding(path):

    # Get annotation path from image path

    path = ANNOTATION_DIR / path.parent.name / path.stem

    

    # Parse boundings

    tree = ET.parse(path)

    bndbox = tree.getroot().findall('object')[0].find('bndbox')

    left = int(bndbox.find('xmin').text)

    right = int(bndbox.find('xmax').text) 

    upper = int(bndbox.find('ymin').text)

    lower = int(bndbox.find('ymax').text) 

    

    return (left, upper, right, lower)





def crop_and_save_image(path, save_dir=IMAGES_CROPPED_DIR):

    box = parse_bounding(path)

    

    image = Image.open(path)

    image_cropped = image.crop(box)

    image_cropped = image_cropped.convert('RGB')

    image_cropped.save(save_dir / path.name)
# Crop images according to bounding boxes

try:

    pool = mp.Pool(processes=mp.cpu_count())

    pool.map(crop_and_save_image, all_image_paths)

except Exception as e:

    print(e)

finally:

    pool.close()



    

all_image_cropped_paths = get_all_file_path(IMAGES_CROPPED_DIR, '.jpg') # PosixPath

all_image_cropped_labels = [path_to_label(path) for path in all_image_cropped_paths] # [0,1,2,...]



assert len(all_image_paths)==len(all_image_cropped_paths), 'Numbers of images and cropped images not match! {}!={}'.format(len(all_image_paths), len(all_image_cropped_paths))
# IMAGE_PATH = all_image_paths

# LABEL = all_image_labels

IMAGE_PATH = all_image_cropped_paths

LABEL = all_image_cropped_labels
IMAGE_PATH[:5]
LABEL[:5]
# Label distribution

_ = plt.hist(LABEL, bins=120)

plt.xlabel('Label index')

plt.ylabel('Count')

plt.title('Label Distribution')

plt.show()
# Ramdomly check a dog image from the dataset

dog = np.random.choice(IMAGE_PATH)

print('Random dog: ', BREED_LABEL_TO_NAME[path_to_label(dog)])

Image.open(dog)
x_train, x_test, y_train, y_test = train_test_split(IMAGE_PATH, 

                                                    LABEL,

                                                    test_size=TEST_SIZE, 

                                                    random_state=RANDOM_SEED,

                                                    shuffle=True,

                                                    stratify=LABEL)



print('Train data: ', len(x_train))

print('Test data: ', len(x_test))
def augmentation(image, label=None):

    image = tf.image.random_flip_left_right(image, seed=RANDOM_SEED)

    image = tf.image.random_brightness(image, max_delta=0.1, seed=RANDOM_SEED)

    image = tf.image.random_contrast(image, lower=0.9, upper=1.1, seed=RANDOM_SEED)

    if label is None:

        return image

    return image, label





def preprocess_image(image):

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, [INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE])

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image





def load_and_preprocess_image(path):

    image = tf.io.read_file(path)

    return preprocess_image(image)





def load_and_preprocess_from_path_label(path, label):

    return load_and_preprocess_image(path), label
ds_train = tf.data.Dataset.from_tensor_slices(([str(path) for path in x_train], y_train))

ds_test = tf.data.Dataset.from_tensor_slices(([str(path) for path in x_test], y_test))



# Apply shuffle and repeat on training data

ds_train = ds_train.apply(

    tf.data.experimental.shuffle_and_repeat(buffer_size=len(x_train), seed=RANDOM_SEED))



# Preprocessing

ds_train = ds_train.map(load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)

ds_test = ds_test.map(load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)



# Augmentation

# ds_train = ds_train.map(augmentation, num_parallel_calls=AUTOTUNE)



ds_train = ds_train.batch(BATCH_SIZE)

ds_test = ds_test.batch(BATCH_SIZE)



# `prefetch` lets the dataset fetch batches in the background while the model is training.

ds_train = ds_train.prefetch(buffer_size=1)
# Fine tuning based on MobileNetV2

base_model = tf.keras.applications.MobileNetV2(input_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3), include_top=False)

base_model.trainable = False
model = tf.keras.Sequential([

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(len(BREED_NAME_TO_LABEL), 

                          activation='softmax', 

                          kernel_initializer=tf.keras.initializers.glorot_normal(seed=RANDOM_SEED),

                          bias_initializer='zeros',

                          name='predictions')

])



model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),

              loss='sparse_categorical_crossentropy',

              metrics=["accuracy"]

)



model.summary()
# Start training the model

steps_per_epoch = len(x_train)//BATCH_SIZE

history = model.fit(ds_train, epochs=EPOCH, validation_data=ds_test, steps_per_epoch=steps_per_epoch)
# Plot training & validation metrics

def plot_model_history(history):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    fig.suptitle('Model Training Metrics')



    ax1.plot(history.history['accuracy'])

    ax1.plot(history.history['val_accuracy'])

    ax1.title.set_text('Accuracy')

    ax1.set_ylabel('Accuracy')

    ax1.set_xlabel('Epoch')

    ax1.legend(['Train', 'Valid'], loc='upper left')



    ax2.plot(history.history['loss'])

    ax2.plot(history.history['val_loss'])

    ax2.title.set_text('Loss')

    ax2.set_ylabel('Loss')

    ax2.set_xlabel('Epoch')

    ax2.legend(['Train', 'Valid'], loc='upper left')



    fig.show()

    

    

plot_model_history(history)
# # Fine tune layers of the MobileNet base model

# base_model.trainable = True

# fine_tune_at = 152



# # Freeze all the layers before the `fine_tune_at` layer

# for layer in base_model.layers[:fine_tune_at]:

#      layer.trainable = False

        



# model.compile(optimizer=tf.keras.optimizers.Adam(1e-8),

#               loss='sparse_categorical_crossentropy',

#               metrics=["accuracy"]

# )



# model.summary()



# history_finetune = model.fit(ds_train, epochs=EPOCH, validation_data=ds_test, steps_per_epoch=steps_per_epoch)



# plot_model_history(history_finetune)
model_version = 'mobilenet_v2_1.0_224_stanford_dogbreeds'
# Save the keras model

model.save('{}.h5'.format(model_version))
# Convert keras model to .tflite

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

tflite_model = converter.convert()

with open('{}.tflite'.format(model_version), 'wb') as f:

    f.write(tflite_model)
# Inference 

def decode_prediction(preds, top=3):

    top_indices = preds.argsort()[-top:][::-1]

    result = [(BREED_LABEL_TO_NAME[i], preds[i]) for i in top_indices] # (labels, scores)

    result.sort(key=lambda x: x[1], reverse=True)

    return '\n'.join(['{}: {:.4f}'.format(*item) for item in result])





def inference(image, model, decode=False):

    preds = model.predict(image)[0]

    if decode:

        result = decode_prediction(preds)

        return result

    return preds





# Inference with .tflite model

def tflite_inference(image, model_file, decode=False):

    interpreter = tf.lite.Interpreter(model_file)

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], tf.cast(image, input_details[0]['dtype']))

    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    if decode:

        result = decode_prediction(preds)

        return result    

    return preds





image_example_path = np.random.choice(x_test)

image_example = load_and_preprocess_image(str(image_example_path))

image_example = tf.expand_dims(image_example, axis=0)

label = BREED_LABEL_TO_NAME[path_to_label(image_example_path)]

print('Label: {}'.format(label))



print('Prediction (keras):')

preds = inference(image_example, model, decode=True)

print(preds)



print('Prediction (tflite):')

preds = tflite_inference(image_example, '{}.tflite'.format(model_version), decode=True)

print(preds)
fig, axes = plt.subplots(5, 4, figsize=(20, 16))

axes = axes.ravel()



for i, ax in enumerate(axes):

    # Randomly test a sample

    image_example_path = np.random.choice(x_test)

    image_example = load_and_preprocess_image(str(image_example_path))

    image_example = tf.expand_dims(image_example, axis=0)

    

    label = BREED_LABEL_TO_NAME[path_to_label(image_example_path)]

    preds = inference(image_example, model, decode=True)



    image = Image.open(image_example_path)

    ax.imshow(image)

    ax.set_title(preds)

    ax.set_xlabel(label)

    ax.grid(False)

        

plt.tight_layout()
preds_test = model.predict(ds_test)

y_preds_test = preds_test.argmax(axis=1)
# Get confusion matrix

conf_mat = confusion_matrix(y_test, y_preds_test)

# np.fill_diagonal(conf_mat, 0)

plt.figure(figsize=(8, 8))

plt.plot(figsize=())

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('True')

plt.imshow(conf_mat)

plt.axis('scaled')

plt.show()
# Get mismatched predictions

row_idx, col_idx = conf_mat.nonzero()

value_count = conf_mat[row_idx, col_idx]

df_conf_mat = pd.DataFrame({'label': row_idx, 'pred': col_idx, 'count': value_count})



df_conf_mat = df_conf_mat.sort_values('count', ascending=False)

df_label_count = df_conf_mat.groupby('label')['count'].sum().to_frame().reset_index()

df_label_count = df_label_count.rename(columns={'count': 'total'})



df_conf_mat = df_conf_mat.merge(df_label_count, how='left', on='label')

df_conf_mat['ratio'] = df_conf_mat['count'] / df_conf_mat['total']



df_conf_mat = df_conf_mat[df_conf_mat['label']!=df_conf_mat['pred']]

df_conf_mat[:10]
def plot_random_image_of_a_breed(image_paths, breed_label, image_num=8):

    breed_code = BREED_LABEL_TO_CODE[breed_label]

    sample_image_of_a_breed_path = [path for path in image_paths if breed_code in str(path)][:image_num]



    fig, axes = plt.subplots(image_num//4, 4, figsize=(14, 8))

    axes = axes.ravel()



    for i, (ax, image_path) in enumerate(zip(axes, sample_image_of_a_breed_path)):

        image = Image.open(image_path)

        ax.imshow(image)

        ax.set_title('{}'.format(image_path.stem))

        ax.grid(False)

        ax.axis('off')



    fig.suptitle(BREED_LABEL_TO_NAME[breed_label])

    fig.tight_layout()
# The `label` in first row

plot_random_image_of_a_breed(x_test, df_conf_mat['label'].iloc[0])
# The `pred` in first row

plot_random_image_of_a_breed(x_test, df_conf_mat['pred'].iloc[0])