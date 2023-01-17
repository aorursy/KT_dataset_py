from keras import layers,models,optimizers
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
train_dir = Path('../input/training/training')
test_dir = Path('../input/validation/validation')
# height and width should be 224 as we are going to use vgg16 and its input shape is 224*224*3
height=224
width=224
channels=3
batch_size=32
seed=99
train_batches=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True).flow_from_directory(train_dir,
                                                         target_size = (height, width),
                                                         batch_size = batch_size,
                                                         seed = 2,
                                                         class_mode = "categorical")
validation_batches=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True).flow_from_directory(test_dir,
                                                              target_size = (height, width),
                                                              batch_size = batch_size,
                                                              seed = 2,
                                                              class_mode = "categorical")
vgg = applications.VGG16(weights = "imagenet")
model = models.Sequential()
for layer in vgg.layers:
    model.add(layer)
model.layers.pop()
#removing the last layer as it is classifying  1000 classes
for layer in model.layers:
    layer.trainable = False
#last layer for classifying 10 classes of monkeys
model.add(layers.Dense(10, activation = "softmax"))
model.compile(optimizers.Adam(lr = 0.009), loss = "categorical_crossentropy", metrics = ["accuracy"])
model.fit_generator(train_batches,
                    steps_per_epoch = 1097//batch_size,
                    validation_data = validation_batches,
                    validation_steps = 4,
                    epochs = 5,
                    verbose  = 2)
#you can fine tune the epochs and other hyperparams to get proper accuracy.
