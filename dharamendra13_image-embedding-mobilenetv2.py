# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base_dir = '/kaggle/input/small-dataset-glrv/small_dataset_glrv2'
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')

BATCH_SIZE = 32
IMG_SIZE = (160,160)

train_dataset = image_dataset_from_directory(train_dir,
                                            shuffle= True,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)
validation_dataset = image_dataset_from_directory(validation_dir,
                                            shuffle= True,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)
class_names = train_dataset.class_names
class_names
plt.figure(figsize=(10,10))
for images,labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('Off')
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size = AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
])
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10,10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,0))
        plt.imshow(augmented_image[0]/255)
        plt.axis('Off')
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1/127.5, offset = -1)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape= IMG_SHAPE,
                                              include_top=False,
                                              weights='imagenet')
base_model.summary()
image_batch,label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(len(class_names))
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
inputs = tf.keras.Input(shape=(160,160,3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs,outputs)
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr= base_learning_rate),
             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
             metrics=['accuracy'])
model.summary()
len(model.trainable_variables)
initial_epochs = 10
loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
model.summary()

len(model.trainable_variables)

fine_tune_epochs = 50
total_epochs =  initial_epochs + fine_tune_epochs

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "/kaggle/working/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         callbacks=[cp_callback],
                         validation_data=validation_dataset)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

image_batch,label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
predictions = tf.nn.softmax(predictions)
print('Predictions:\n', np.argmax(predictions.numpy(),axis=1))
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[np.argmax(predictions[i])])
    plt.axis("off")
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
model.save('mobilenetv2.h5')
model.summary()
from tensorflow.keras.models import Model
intermediate_layer_model = Model(inputs=model.input, 
                                              outputs=model.get_layer("global_average_pooling2d").output)
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50

def get_vec(image_path,intermediate_layer_model):
    """ Gets a vector embedding from an image.
    :param image_path: path to image on filesystem
    :returns: numpy ndarray
    """

    img = image.load_img(image_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet50.preprocess_input(x)
    intermediate_output = intermediate_layer_model.predict(x)

    return intermediate_output[0]
from sklearn.metrics.pairwise import cosine_similarity
import glob
from tqdm.autonotebook import tqdm
tqdm.pandas()
l113209_1_vec = get_vec('/kaggle/input/small-dataset-glrv/small_dataset_glrv2/validation/113209/03af458772602abb.jpg',intermediate_layer_model)
l113209_2_vec = get_vec('/kaggle/input/small-dataset-glrv/small_dataset_glrv2/validation/113209/05b418379cee5736.jpg',intermediate_layer_model)
l126637_1_vec = get_vec('/kaggle/input/small-dataset-glrv/small_dataset_glrv2/validation/126637/023489b87edcd1b0.jpg',intermediate_layer_model)
l126637_2_vec = get_vec('/kaggle/input/small-dataset-glrv/small_dataset_glrv2/validation/126637/02d7841f33e7297b.jpg',intermediate_layer_model)

X = np.stack([l113209_1_vec, l113209_2_vec, l126637_1_vec, l126637_2_vec])
Y = X
similarity_matrix = cosine_similarity(X, Y)

print(similarity_matrix)
image_paths = glob.glob('/kaggle/input/small-dataset-glrv/small_dataset_glrv2/validation/*/*.jpg')
len(image_paths)
image_vectors = {}
for image_path in tqdm(image_paths):
    vector = get_vec(image_path,intermediate_layer_model)
    image_vectors[image_path] = vector
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
X = np.stack(list(image_vectors.values()))
pca_100 = PCA(n_components=20)
pca_result_100 = pca_100.fit_transform(X)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_100.explained_variance_ratio_)))
print(np.shape(pca_result_100))

    
tsne = TSNE(n_components=2, verbose=1, n_iter=3000)
tsne_result = tsne.fit_transform(pca_result_100)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
plt.scatter(tsne_result_scaled[:,0], tsne_result_scaled[:,1])
images = []
for image_path in image_paths:
    image = cv2.imread(image_path, 3)
    b,g,r = cv2.split(image)           # get b, g, r
    image = cv2.merge([r,g,b])         # switch it to r, g, b
    image = cv2.resize(image, (50,50))
    images.append(image)    
fig, ax = plt.subplots(figsize=(20,15))
artists = []

for xy, i in zip(tsne_result_scaled, images):
    x0, y0 = xy
    img = OffsetImage(i, zoom=.7)
    ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
    artists.append(ax.add_artist(ab))
ax.update_datalim(tsne_result_scaled)
ax.autoscale(enable=True, axis='both', tight=True)
plt.show()
intermediate_layer_model.save('intermediate_layer_model.h5')
