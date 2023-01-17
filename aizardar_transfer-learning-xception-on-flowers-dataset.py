import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

dataset, info = tfds.load("tf_flowers", as_supervised = True, with_info = True)

# By setting with_info = True, we get information about our dataset
info.splits
dataset_size = info.splits["train"].num_examples
print(dataset_size)
# Let's also checkout number of classes and their names

class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
print(class_names, n_classes)
# We will use TF Datasets API for splitting

test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
    as_supervised=True)
import matplotlib.pyplot as plt 

plt.figure(figsize =(12,10))
index = 0
for image, label in train_set_raw.take(9):
    index +=1
    plt.subplot(3,3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")
plt.show()
def preprocess(image, label):
    resized_image = tf.image.resize(image,[224,224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label
batch_size = 32
train_set = train_set_raw.shuffle(1000)
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1) # Prefetch one batch to make sure that a batch is ready to be served at all time
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)
base_model = keras.applications.xception.Xception(weights="imagenet", include_top = False)

# We will now add our own global average pooling layer followed by a dense output layer with one 
# unit per class using the softmax activation function.

avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation = "softmax")(avg)
model = keras.Model(inputs = base_model.input, outputs = output)
for index, layer in enumerate(base_model.layers):
    print(index,layer.name)
    
for layer in base_model.layers:
    layer.trainable = False
optimizer = keras.optimizers.SGD(lr=0.2, momentum = 0.9, decay = 0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
history = model.fit(train_set, epochs = 10, validation_data = valid_set)
for layer in base_model.layers:
    layer.trainable = True
optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.9, decay = 0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set, epochs = 50, validation_data = valid_set)