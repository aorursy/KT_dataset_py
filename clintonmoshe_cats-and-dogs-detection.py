import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D,Flatten,Dropout,Activation
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
tf.config.experimental.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
batch_size = 64
epochs = 20
image_height = 200
image_width = 200
cats = plt.imread('../input/dogs-cats-images/dataset/test_set/cats/cat.4020.jpg')
dogs = plt.imread('../input/dogs-cats-images/dataset/test_set/dogs/dog.4014.jpg')

class_names = ['cat','dog']

plt.subplot(121)
plt.imshow(cats)
plt.xlabel(class_names[0])
plt.subplot(122)
plt.imshow(dogs)
plt.xlabel(class_names[1])
plt.grid(False)
    
plt.show()
print('Cats data shape: {}'.format(cats.shape))
print('Cats data shape: {}'.format(dogs.shape))
print('Cats min value: {}'.format(cats.min()))
print('cats max value: {}'.format(cats.max()))
train_directory = '../input/dogs-cats-images/dataset/training_set'
test_directory = '../input/dogs-cats-images/dataset/test_set'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.1
)

validation_datagen = ImageDataGenerator(
    rescale=1./255
)
train_data = train_datagen.flow_from_directory(
                train_directory,
                shuffle=True,
                target_size=(image_height, image_width),
                batch_size=batch_size,
                class_mode='binary'
)

test_data = validation_datagen.flow_from_directory(
                test_directory,
                target_size=(image_height, image_width),
                batch_size=batch_size,
                class_mode='binary'
)
sample_train_images, _ = next(train_data)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plotImages(sample_train_images[:5])
model = Sequential([
    Conv2D(32, 3, activation='relu', padding='same',kernel_initializer='he_uniform',input_shape=(image_height, image_width ,3)),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu', padding='same',kernel_initializer='he_uniform',),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu', padding='same',kernel_initializer='he_uniform',),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu',kernel_initializer='he_uniform'),
    Dense(1, activation='sigmoid')
])
model.summary()
model.compile(optimizer=SGD(lr=0.001, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(len(test_data))
history = model.fit_generator(
    train_data,
    steps_per_epoch = len(train_data),
    epochs=epochs,
    validation_data=test_data,
    validation_steps= len(test_data)
)
_, acc = model.evaluate_generator(test_data, steps=len(test_data), verbose=0)
print('accuracy: {:.2f}'.format(acc * 100))
def make_prediction(image_path):
    img = image.load_img(image_path, target_size = (200, 200))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict_classes(img)
    # class 1 is for cats, 0 is for dogs
    #return predictions
    if predictions == 0:
         return print('cat', predictions[0][0])
    else:
         return print('dog', predictions[0][0])
    
cat = '../input/dogs-cats-images/dataset/test_set/cats/cat.4020.jpg'
dog = '../input/dogs-cats-images/dataset/test_set/dogs/dog.4014.jpg'
make_prediction(dog)
def get_model():
    model = Sequential([
    Conv2D(32, 3, activation='relu', padding='same',kernel_initializer='he_uniform',input_shape=(image_height, image_width ,3)),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu', padding='same',kernel_initializer='he_uniform',),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu', padding='same',kernel_initializer='he_uniform',),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu',kernel_initializer='he_uniform'),
    Dense(1, activation='sigmoid')
])
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model
    
model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
model_vgg = VGG19(weights='imagenet', include_top=False, input_shape=(200,200,3))
x = model_vgg.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
pred = Dense(1, activation='sigmoid')(x)

new_model = Model(inputs=model_vgg.input, outputs=pred)

for layer in model_vgg.layers:
    layer.trainable = False
    
new_model.compile(optimizer=SGD(lr=0.001, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])

new_model.fit_generator(train_data,
    steps_per_epoch = len(train_data),
    epochs=10,
    validation_data=test_data,
    validation_steps= len(test_data))

_, acc = new_model.evaluate_generator(test_data, steps=len(test_data), verbose=0)
print('accuracy: {:.2f}'.format(acc * 100))


model_json = new_model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
new_model.save_weights("final_model.h5")
print("Saved model to output folder")
# class 0 is cat, 1 is dog
def make_prediction_vgg19(image_path):
    img = image.load_img(image_path, target_size = (200, 200))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predictions = new_model.predict(img)
    return predictions
cat = '../input/dogs-cats-images/dataset/test_set/cats/cat.4007.jpg'
dog = '../input/dogs-cats-images/dataset/test_set/dogs/dog.4018.jpg'
make_prediction_vgg19(cat)