!pip install tensorflow_datasets
!pip install -q git+https://github.com/tensorflow/docs
!pip install tensorflowjs
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import itertools

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import tensorflow_docs as tfdocs
import tensorflow_docs.plots

AUTOTUNE = tf.data.experimental.AUTOTUNE
tfds.disable_progress_bar()

print('Using TensorFlow Version:', tf.__version__)
splits = ['train[:80%]', 'train[80%:]', 'test']

dataset, info = tfds.load(name='rock_paper_scissors', with_info=True, as_supervised=True, split=splits)

(train_examples, validation_examples, test_examples) = dataset
num_examples = info.splits['train'].num_examples + info.splits['test'].num_examples
num_classes = info.features['label'].num_classes

print('The dataset has a total of:')
print('\u2022 {:,} classes'.format(num_classes))
print('\u2022 {:,} images'.format(num_examples))
class_names = ['rock', 'paper', 'scissors']
def plot_image(examples):
    for image, label in examples:
      image = image.numpy()
      label = label.numpy()

      plt.imshow(image)
      plt.show()
      print('The label of this image is:', label)
      print('The class name of this image is:', class_names[label])

plot_image([(image, label) for (image, label) in train_examples.take(1)])
"""
model_selection = ("mobilenet_v2", 224, 1280) #@param ["(\"mobilenet_v2\", 224, 1280)", "(\"inception_v3\", 299, 2048)"] {type:"raw", allow-input: true}

handle_base, pixels, FV_SIZE = model_selection

IMAGE_SIZE = (pixels, pixels)

MODULE_HANDLE ="https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)

feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                   input_shape=IMAGE_SIZE + (3,))

print("Using {} with input size {} and output dimension {}.".format(handle_base, IMAGE_SIZE, FV_SIZE))
"""
IMAGE_SIZE = (224, 224)

feature_extractor = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE + (3,),
                                               include_top=False,
                                               weights='imagenet')
def format_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    image = tf.cast(image, dtype=tf.float32)
    return  image, label

def augment_image(image, label):
    image, label = format_image(image, label)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return image, label

BATCH_SIZE =  32

train_batches = train_examples.shuffle(num_examples // 4).map(augment_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
validation_batches = validation_examples.map(augment_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_batches = test_examples.map(format_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
plot_image([(image, label) for (image, label) in train_batches.unbatch().take(1)])
model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Flatten(),                        # Comment out if using a feature_extractor from TF Hub
        tf.keras.layers.Dense(1024, activation='relu'),   # Comment out if using a feature_extractor from TF Hub
        tf.keras.layers.Dropout(0.5),                     # Comment out if using a feature_extractor from TF Hub
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()
do_fine_tuning = False #@param {type:"boolean"}

if do_fine_tuning:
    feature_extractor.trainable = True    
else:
    feature_extractor.trainable = False
if do_fine_tuning:
    optimizer=tf.keras.optimizers.SGD(lr=0.002, momentum=0.9)
else:
    optimizer = 'adam'
    
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
EPOCHS = 10

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(train_batches,
                    epochs=EPOCHS,
                    callbacks=[callback],
                    validation_data=validation_batches)
plotter = tfdocs.plots.HistoryPlotter()
plotter.plot({"History": history}, metric="accuracy")
plt.title("Accuracy")
plt.ylim([0.5,1])
plotter = tfdocs.plots.HistoryPlotter()
plotter.plot({"History": history}, metric="loss")
plt.title("Loss")
plt.ylim([0,7])
eval_results = model.evaluate(test_batches, verbose=0)

for metric, value in zip(model.metrics_names, eval_results):
    print(metric + ': {:.4}'.format(value))
def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      color = "white" if cm[i, j] > threshold else "black"
      plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
test_data = list(test_batches.unbatch())

test_images = []
test_labels = []
for image, label in test_data:
    test_images.append(image.numpy())
    test_labels.append(label.numpy())

test_images = np.array(test_images)
test_labels = np.array(test_labels)
test_pred_raw = model.predict(test_images)
test_pred = np.argmax(test_pred_raw, axis=1)

cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
plot_confusion_matrix(cm, class_names=class_names)
saved_model_path = "./rps_model.h5"

model.save(saved_model_path)
!tensorflowjs_converter --input_format=keras {saved_model_path} ./