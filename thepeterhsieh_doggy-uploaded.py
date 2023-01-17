from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import cv2 
import numpy as np
import pandas as pd
train_dir = '../input/aia-st4-cnn-identifying-dogs/training_set/training_set'
test_dir = '../input/aia-st4-cnn-identifying-dogs/testing_set/testing_set'

# print traing data and testing data
## len of the glob's returned list
print("training data number：", len(glob(train_dir + '/*/*')))
print("testing data nummber："  , len(glob(test_dir + '/test/*')))

# check the images
img = cv2.imread(train_dir+"/basset/100.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(type(img), img.shape, img.min(), img.max())
image_size = 64
batch_size = 32
seed = 248

# Data augmentation and split the data into valid and training into batches 

# create a image generator object
## to rescale, train_test_split, rotate, shift, flip, ...
## Q: how many of the augmented data could be generated?
train_datagen = ImageDataGenerator(
    rescale= 1./255,
    validation_split=0.1,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# feed the data to the image generator object
## set the image to read data, convert to rgb, resize, shuffle with a seed, pack into batch size 
### return batches of the input data with augemntation
train_generator = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size = (image_size,image_size), # resize
    color_mode='rgb',
    batch_size = batch_size,
    class_mode = 'categorical', # 2D one-hot encode 
    seed=seed,
    subset = 'training'
)

validation_generator = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed,
    subset = 'validation'
)


# return a Directory Iterator object 
print(type(train_generator))
# print the labels 
print(train_generator.class_indices)
# print the samples of train and validation
print(train_generator.labels, validation_generator.samples)
## Q: where to find the attributes?
from tensorflow.keras import Input, layers, models, optimizers, Model
from tensorflow.keras.callbacks import ModelCheckpoint
def network_0(input_image):
    # first block of conv layers
    ## Conv(3x3x32) -> Conv(3x3x32) -> ReLU -> BatchNorm -> MaxPool
    ### Q: the sequence matters?
    x = layers.Conv2D(filters=32, kernel_size=3, padding='same')(input_image)
    x = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x) # stride default will be the same as the pool_size
    
    # second block of conv layers
    ## Conv(3x3x64) -> Conv(3x3x64) -> ReLU -> BatchNorm -> MaxPool
    x = layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # third block of conv layers
    ## Conv(3x3x128) -> Conv(3x3x128) -> ReLU -> BatchNorm -> MaxPool
    x = layers.Conv2D(filters=128, kernel_size=3, padding='same')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D(pool_size=3)(x)
    
    # fourth, fully dense layer
    ## flatten -> drop -> dense
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    prob = layers.Dense(7, activation='softmax')(x) # softmax for multi-classification probability
    
    # Q: keras.Model or models.Model
    output = Model(input_image, prob)
    
    return output
# recommend using keras.Input instead of keras.layers.InputLayer
## a placeholder tensor, shape is withoiut the batch axis (only the shape of the input image)
### Q: keras.Input or layers.Input
input_image = Input(shape=(image_size, image_size, 3))

# initialize the traninable weights and other constraints like regularizer
network = network_0(input_image)
network.summary()
# why batchnormaliztion has parameter = 4 * channels?
# where does the non-trainable params come from?
epochs=30

# optimizer
opt = optimizers.Adam(learning_rate=1e-3, decay=1e-6)

# checkpoint
## 0_temp.h5, 1_temp.h5, ...
mcp_fpath ='./weights/baseline/0_temp.h5'
# or each best and store as 
## mcp_fpath = './weights/baseline/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
mcp = ModelCheckpoint(filepath=mcp_fpath, 
                      save_best_only=True,
                      save_weights_only=True, # for load_weights()
                      monitor='val_loss', 
                      mode='auto')

# compile 
network.compile(loss='categorical_crossentropy', 
                optimizer=opt, 
                metrics=['accuracy'])

# training and history
## pass the DirectoryIterator as the args
## then no need to specify y, and batch_size (already in the Iterator)
training = network.fit(x=train_generator, 
                      epochs=epochs,
                      steps_per_epoch=train_generator.samples/train_generator.batch_size, 
                      validation_data=validation_generator,
                      validation_steps=validation_generator.samples/validation_generator.batch_size,
                      verbose=2,
                      callbacks=[mcp]) # list of the CallBack instances
import matplotlib.pyplot as plt

print(training.history.keys())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title('model '+ 'loss')
plt.plot(np.arange(1, len(training.history['loss'])+1), training.history['loss'], label='loss')
plt.plot(np.arange(1, len(training.history['loss'])+1), training.history['val_loss'], label='val_loss')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.title('model '+ 'acc')
plt.plot(np.arange(1, len(training.history['loss'])+1), training.history['accuracy'], label='accuracy')
plt.plot(np.arange(1, len(training.history['loss'])+1), training.history['val_accuracy'], label='val_accuracy')
plt.legend(loc='best')

plt.show()
# initialize the model weights with the callbacks 
#network.load_weights(mcp_fpath)
# load the test data
## normalize it before input the data to save space
test_datagen = ImageDataGenerator(
    rescale= 1./255)

test_generator = test_datagen.flow_from_directory(
    directory = test_dir,
    target_size=(image_size,image_size),
    batch_size=batch_size,
    class_mode=None, # only for load the data, no y
    shuffle=False)
# predict
## with steps=None, it will run till the data is exhausted
pred = network.predict(x=test_generator, verbose=1, steps=None)

## with the categorical pred after softmax, 
## the max will be the best prediction
pred = np.argmax(pred,axis=1)
print(pred.shape)
print(pred)
# map to the required classmap
classmap = pd.read_csv('../input/aia-st4-cnn-identifying-dogs/classmap.csv',header=None)
classmap = dict(classmap.values)
pred_class = list(train_generator.class_indices.keys())
print(classmap, end='\n\n')
print(pred_class)

pred_classmap = np.array([classmap[pred_class[i]] for i in pred])
print(pred_classmap)
# save the prediction to a csv file
fname = [f[5:15] for f in test_generator.filenames]

print(type(fname), type(pred_classmap))

result = pd.DataFrame({"id": fname,
                      "class":pred_classmap})
print(result.head(10))

result.to_csv('./result.csv',index=None)

