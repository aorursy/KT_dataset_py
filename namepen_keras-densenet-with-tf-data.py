import os

import time

import math



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



import tensorflow as tf

#tf.enable_eager_execution()

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.densenet import preprocess_input



from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
df = pd.read_csv('../input/humpback-whale-identification/train.csv')

df.head()
df.count()
def prepare_labels(y):

    values = np.array(y)

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(values)

    # print(integer_encoded)



    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # print(onehot_encoded)



    y = onehot_encoded

    print(y.shape)

    return y, label_encoder
y, label_encoder = prepare_labels(df['Id'])
labels_count = df.Id.value_counts()
new_train = pd.DataFrame(columns=['Image', 'Id'])
train_names = df.index.values



dup = []

for idx,row in df.iterrows():

    if labels_count[row['Id']] < 4:

        dup.extend([idx]*math.ceil((4 - labels_count[row['Id']])/labels_count[row['Id']]))

    if idx == 25360:

        print('last class')

        

train_names = np.concatenate([train_names, dup])

train_names = train_names[np.random.RandomState(seed=42).permutation(train_names.shape[0])]

len(train_names)
count = 0

for i in range(len(train_names)):

    new_train = new_train.append(df.loc[[train_names[i]]])

    count +=1

print(new_train.count())
del df

del y, label_encoder

del train_names
new_y, label_encoder = prepare_labels(new_train['Id'])
new_train.to_csv('new_train.csv')
#new_train = pd.read_csv('../input/kernelbe655f6ff1/new_train.csv')

#new_train.head()

#tr_y, tr_label_encoder = prepare_labels(new_train['Id'])
def load_image(path):

    path='../input/humpback-whale-identification/train/' + path

    image_string = tf.read_file(path)



    # Don't use tf.image.decode_image, or the output shape will be undefined

    image = tf.image.decode_jpeg(image_string, channels=3)



    # This will convert to float values in [0, 1]

    image = tf.image.convert_image_dtype(image, tf.float32)



    image = tf.image.resize_images(image, [224, 224])

    return image
def flip(x: tf.Tensor) -> tf.Tensor:

    """Flip augmentation



    Args:

        x: Image to flip



    Returns:

        Augmented image

    """

    x = tf.image.random_flip_left_right(x)

    #x = tf.image.random_flip_up_down(x)



    return x



def color(x: tf.Tensor) -> tf.Tensor:

    """Color augmentation



    Args:

        x: Image



    Returns:

        Augmented image

    """

    x = tf.image.random_hue(x, 0.08)

    x = tf.image.random_saturation(x, 0.6, 1.6)

    x = tf.image.random_brightness(x, 0.05)

    x = tf.image.random_contrast(x, 0.7, 1.3)

    return x



def rotate(x: tf.Tensor) -> tf.Tensor:

    """Rotation augmentation



    Args:

        x: Image



    Returns:

        Augmented image

    """



    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=-1, maxval=1, dtype=tf.int32))



def zoom(x: tf.Tensor) -> tf.Tensor:

    """Zoom augmentation



    Args:

        x: Image



    Returns:

        Augmented image

    """



    # Generate 20 crop settings, ranging from a 1% to 20% crop.

    scales = list(np.arange(0.8, 1.0, 0.01))

    boxes = np.zeros((len(scales), 4))



    for i, scale in enumerate(scales):

        x1 = y1 = 0.5 - (0.5 * scale)

        x2 = y2 = 0.5 + (0.5 * scale)

        boxes[i] = [x1, y1, x2, y2]



    def random_crop(img):

        # Create different crops for an image

        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(224,224))

        # Return a random crop

        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]





    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)



    # Only apply cropping 50% of the time

    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))
train_data = tf.data.Dataset.from_tensor_slices((new_train.Image, new_y))
train_data = train_data.map(lambda x,y: (load_image(x),y))
# sample x_2 data

augmentations = [flip, zoom, rotate]



for f in augmentations:

    train_data = train_data.map(lambda x,y: (tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: f(x), lambda: x), y))

train_data = train_data.map(lambda x,y : (tf.clip_by_value(x, 0, 1), y))
train_data = train_data.batch(32).repeat()

train_data = train_data.prefetch(1)
from tensorflow.keras.applications.densenet import DenseNet121
#load the base densenet model

model = DenseNet121(include_top=True, weights=None, input_shape=(224,224,3), classes=5005)
#Add the metrics

'''

def top_5_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=5)

'''
model.compile(optimizer=Adam(lr=3e-4), loss='categorical_crossentropy',

              metrics=[categorical_crossentropy, categorical_accuracy])

print(model.summary())
start = time.time()

history = model.fit(train_data, steps_per_epoch=len(new_train)//32, epochs=5, verbose=1)



print("Finish Training : {}".format(time.time()-start))
test = os.listdir("../input/humpback-whale-identification/test/")

print(len(test))
col = ['Image']

test_df = pd.DataFrame(test, columns=col)

test_df['Id'] = ''
#For prediction

dumy_y = np.zeros([7960,5005])
def test_load_image(path):

    path='../input/humpback-whale-identification/test/' + path

    image_string = tf.read_file(path)



    # Don't use tf.image.decode_image, or the output shape will be undefined

    image = tf.image.decode_jpeg(image_string, channels=3)



    # This will convert to float values in [0, 1]

    image = tf.image.convert_image_dtype(image, tf.float32)



    image = tf.image.resize_images(image, [224, 224])

    return image
test_data = tf.data.Dataset.from_tensor_slices((test_df.Image, dumy_y))

test_data = test_data.map(lambda x,y : (test_load_image(x),y))

test_data = test_data.batch(32).repeat()
predictions = model.predict(test_data, steps=7960//32, verbose=1)
for i, pred in enumerate(predictions):

    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_df.head(10)

test_df.to_csv('submission.csv', index=False)