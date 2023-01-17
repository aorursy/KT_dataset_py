
import os
print(os.listdir("../input/cars_train"))


from numpy.random import seed
seed(17)
from tensorflow import set_random_seed
set_random_seed(42)
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 2 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

img_width, img_height = 299, 299

path='../input/cars_train/'
train_data_dir = path + 'trein'
validation_data_dir = path + 'valid'
test_data_dir = path + 'test'

# hyper parameters for model
nb_classes = 2  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
img_width, img_height = 299, 299  # change based on the shape/structure of your images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
nb_train_samples = 170  # Total number of train samples. NOT including augmented images
nb_validation_samples = 39  # Total number of train samples. NOT including augmented images.


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)

predictions = Dense(nb_classes, activation='softmax', name='predictions')(x)

# add your top layer block to your base model
model = Model(base_model.input, predictions)
#print(model.summary())

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all layers of the based model that is already pre-trained.
for layer in base_model.layers:
    layer.trainable = False

# Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
# To save augmentations un-comment save lines and add to your flow parameters.
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=transformation_ratio,
                                       shear_range=transformation_ratio,
                                       zoom_range=transformation_ratio,
                                       cval=transformation_ratio,
                                       horizontal_flip=True,
                                       vertical_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=[img_width, img_height],
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=[img_width, img_height],
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])

# Train Simple CNN
model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=nb_epoch / 5,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size)

# based_model_last_block_layer_number points to the layer in your model you want to train.
# For example if you want to train the last block of a 19 layer VGG16 model this should be 15
# If you want to train the last Two blocks of an Inception model it should be 172
# layers before this number will used the pre-trained weights, layers above and including this number
# will be re-trained based on the new data.
for layer in model.layers[:based_model_last_block_layer_number]:
    layer.trainable = False
for layer in model.layers[based_model_last_block_layer_number:]:
    layer.trainable = True

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    
# fine-tune the model
history = model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=nb_epoch,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size)




# plot history
loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
if len(loss_list) == 0:
    print('Loss is missing in history')
       
## As loss always exists
epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
## Loss
import matplotlib.pyplot as plt

plt.figure(1)
for l in loss_list:
    plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
for l in val_loss_list:
    plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
    
## Accuracy
plt.figure(2)
for l in acc_list:
    plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
for l in val_acc_list:    
    plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# test model
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = validation_datagen.flow_from_directory(test_data_dir,
                                                                  target_size=[img_width, img_height],
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1,steps=5)
score = model.evaluate_generator(generator=test_generator, steps=5)

print('Test accuracy:', score[1])
