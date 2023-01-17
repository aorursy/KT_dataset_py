import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
# Paths to training and testing csv files
ROOT = "../input/sign-language-mnist/"
test_path = ROOT + "sign_mnist_test/sign_mnist_test.csv"
train_path = ROOT + "sign_mnist_train/sign_mnist_train.csv"
# Visulaising the training data column headings
with open(train_path) as file:
    data = csv.reader(file)
    for i, row in enumerate(data):
        if i == 0:
            print(row)
# Extracts data from csv files
def get_data(file_path):
    with open(file_path) as file:
        data = csv.reader(file)
        labels = []
        images = []
        for i, row in enumerate(data):
            if i == 0:
                continue
            labels.append(row[0])
            image = row[1:]
            image_array = np.array_split(image, 28)
            images.append(image_array)
    return np.array(images).astype(float), np.array(labels).astype(float)

# Extract data from train and test csv files
train_images, train_labels = get_data(train_path)
test_images, test_labels = get_data(test_path)

# Create validation data set
split = 0.8
train_split = int(split*len(train_labels))
validation_images = train_images[train_split:]
validation_labels = train_labels[train_split:]
train_images = train_images[:train_split]
train_labels = train_labels[:train_split]
print("There are {} training images of shape {} by {} with {} labels. There are {} validation images of shape {} by {} with {} labels. There are {} test images of shape {} by {} with {} labels".format(train_images.shape[0], train_images.shape[1], train_images.shape[2], train_labels.shape[0], validation_images.shape[0], validation_images.shape[1], validation_images.shape[2], validation_labels.shape[0], test_images.shape[0], test_images.shape[1], test_images.shape[2], test_labels.shape[0]))
# Returns a list of unique labels and number of occurances for each
def counter(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return list(unique.astype(int)), list(counts)

unique_train, counts_train = counter(train_labels)
unique_test, counts_test = counter(test_labels)

# Plotting bar chart of the counts for training and testing labels
# bar locations and width
x = np.arange(len(unique_train))  
width = 0.35  

fig, ax = plt.subplots(figsize=(12,6))
test = ax.bar(x - width/2, counts_train, width, label='Train')
train = ax.bar(x + width/2, counts_test, width, label='Test')

# Add text for labels, title and x-axis tick labels
ax.set_ylabel('Data Count', fontsize=12)
ax.set_xlabel('Sign', fontsize=12)
ax.set_title('Quantity of Classes In The Test And Train Sets', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(unique_train)
ax.legend()

fig.tight_layout()

plt.show()
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
[axi.set_axis_off() for axi in axs.ravel()]
for i, image in enumerate(train_images[:16]):
    a = fig.add_subplot(4, 4, i + 1)
    plt.imshow(image)
    a.set_title(train_labels[i])

plt.show()
# Converting image dimensions to (X, 28, 28, 1)
train_images = np.expand_dims(train_images, axis=3)
validation_images = np.expand_dims(validation_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# Training ImageGenerator
train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

train_gen = train_datagen.flow(train_images, train_labels, batch_size=32)

# Validation ImageGenerator
valid_datagen = ImageDataGenerator(
    rescale=1/255.)

valid_gen = valid_datagen.flow(validation_images, validation_labels, batch_size=32)

# Testing Image Generator
test_datagen = ImageDataGenerator(
    rescale=1/255.)

test_gen = test_datagen.flow(test_images, test_labels, batch_size=32)
# Choosing an example image for augmentation
example_image = train_images[7:8]
example_label = train_labels[7:8]
example_gen = train_datagen.flow(example_image, example_label, batch_size=1)

# Plotting a grid of the example image after augmentation
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
[axi.set_axis_off() for axi in axs.ravel()]
for i in range (16):
    x_batch, y_batch = next(example_gen)
    a = fig.add_subplot(4, 4, i + 1)
    image = x_batch[0]
    plt.imshow(image[:, :, -1])
    a.set_title(y_batch[0])
plt.show()
# Defining the CNN 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Printing a summary of the model structure
model.summary()
# Fitting the model to the training data
history = model.fit_generator(
    train_gen,
    steps_per_epoch=len(train_images)/32,
    epochs=10, validation_data=valid_gen,
    validation_steps=len(validation_images)/32
)

# Metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting accuracy
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy', fontsize=12)
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch', fontsize=12)

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss', fontsize=12)
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch', fontsize=12)
plt.show()
# Modified Augmentations
train_datagen2 = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=30, #Reduced the rotation range
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    # Removed Flip which might change the meaning of a sign
    fill_mode='nearest')

train_gen2 = train_datagen2.flow(train_images, train_labels, batch_size=64) #Increased batch size

# Modified Model
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(), # New layer
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),  # New layer
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # New layer
    tf.keras.layers.BatchNormalization(),# New layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.1),  # New layer
    tf.keras.layers.Dense(26, activation='softmax')
])

# Adding in a learning rate reducer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# Compiling and printing the model
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.summary()

# Training
history2 = model2.fit_generator(
    train_gen2, 
    steps_per_epoch=len(train_images)/64,
    epochs=15, # Training for longer 
    validation_data=valid_gen,
    validation_steps=len(validation_images)/32, 
    callbacks=[learning_rate_reduction] 
)
acc = history2.history['accuracy']
val_acc = history2.history['val_accuracy']
loss = history2.history['loss']
val_loss = history2.history['val_loss']

# Plotting accuracy
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy', fontsize=12)
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch', fontsize=12)

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss', fontsize=12)
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch', fontsize=12)
plt.show()
print("Evaluate on test data:")
results = model2.evaluate(test_gen, batch_size=len(test_images)/32)
print("test loss, test acc:", results)
# Define a new model that takes an image as input and outputs intermediate representations layers Model2
successive_outputs = [layer.output for layer in model2.layers]
visualization_model = tf.keras.models.Model(inputs = model2.input, outputs = successive_outputs)
# Pass the example image into the model
successive_feature_maps = visualization_model.predict(example_gen)

# Plotting the intermdiate image representations
# Layer names in plot
layer_names = [layer.name for layer in model2.layers]
# Plotting only conv / maxpool/ batchnorm. layers, not Dense layers
np.seterr(divide='ignore', invalid='ignore')
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # number of features in feature map
    n_features = feature_map.shape[-1] 
    # Feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Visualisations
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
