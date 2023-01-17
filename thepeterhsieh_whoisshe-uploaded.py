from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from glob import glob
import cv2 
import numpy as np
import pandas as pd
train_dir = '../input/aia-dt4-who-is-she/training_set/training_set'
test_dir = '../input/aia-dt4-who-is-she/testing_set'

# print traing data and testing data
## len of the glob's returned list
print("training data number：", len(glob(train_dir + '/*/*')))
print("testing data nummber："  , len(glob(test_dir + '/testing_set/*')))

# check the images
img = cv2.imread(train_dir+"/akane/000.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(type(img), img.shape, img.min(), img.max())
image_size = 244 # same as the resnet50 
batch_size = 16
seed = 248

# Data augmentation and split the data into valid and training into batches 

# create a image generator object
## to rescale, train_test_split, rotate, shift, flip, ...
## Q: how many of the augmented data could be generated?
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale= 1./255,
    featurewise_center=False,
    samplewise_center=True,
    featurewise_std_normalization=False,
    samplewise_std_normalization=True,
    zca_whitening=True,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=False,
    horizontal_flip=True)

# feed the data to the image generator object
## set the image to read data, convert to rgb, resize, shuffle with a seed, pack into batch size 
### return batches of the input data with augemntation
train_generator = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size = (image_size,image_size),
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
# print(type(train_generator))

# print the labels 
print(train_generator.class_indices)
from tensorflow.keras import Input, layers, models, optimizers, Model, Sequential, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.applications import ResNet50, ResNet152V2, Xception
## define the customized VGG16
def vgg16CNN(input_shape, outclass, sigma='sigmoid'):
    
    # create a model framework
    model = Sequential()
    
    # base_model: VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # reserve the last five as trainable layers 
    #for layer in base_model.layers[:-5]:
        #layer.trainable=False
    model.add(base_model)
    
    # add the fully-connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer='l1')) # Lasso for feature selection
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer='l1'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(outclass, activation=sigma))
    
    return model
## define the customized resnet50
def CNN(input_shape, outclass, sigma='sigmoid'):
    
    # create a model framework
    model = Sequential()
    
    # base_model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    # reserve the last five as trainable layers 
    for layer in base_model.layers[:-20]:
        layer.trainable=False
    model.add(base_model)
    print(base_model.output_shape)
    
    # add the fully-connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu', kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.1))) # Lasso for feature selection
    model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.1)))
    #model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(256,kernel_regularizer='l2', activation='relu'))
    #model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(256, activation='relu', kernel_regularizer='l1')) # Lasso for feature selection
    #model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(128, activation='relu', kernel_regularizer='l1'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(outclass, activation=sigma))
    
    return model
network = CNN(input_shape=(image_size, image_size, 3),
                  outclass=5, # the number of output labels 
                  sigma='softmax')

print(network.input_shape, network.output_shape)
#network.summary()

'''for layer in network.layers[0].layers:
    print(layer.trainable)'''
# network.load_weights(mcp_fpath)
epochs=20

# optimizer
opt = optimizers.Adam(learning_rate=1e-3)

# checkpoint
## 0_temp.h5, 1_temp.h5, ...
mcp_fpath ='./3_temp.h5'
# or each best and store as 
## mcp_fpath = './weights/baseline/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
mcp = ModelCheckpoint(filepath=mcp_fpath, 
                      save_best_only=True,
                      save_weights_only=False, # for load model
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
network.load_weights(mcp_fpath)
# load the test data
## normalize it before input the data to save space
test_datagen = ImageDataGenerator(
    rescale= 1./255,
    preprocessing_function=preprocess_input)

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
classmap = pd.read_csv('../input/aia-dt4-who-is-she/classmap.csv',header=None)
classmap = dict(classmap.values)
pred_class = list(train_generator.class_indices.keys())
print(classmap, end='\n\n')
print(pred_class)

pred_classmap = np.array([classmap[pred_class[i]] for i in pred])
print(pred_classmap)
# save the prediction to a csv file
fname = [f[12:22] for f in test_generator.filenames]

print(type(fname), type(pred_classmap))

result = pd.DataFrame({"id": fname,
                      "class":pred_classmap})
print(result.head(10))

result.to_csv('./result.csv',index=None)
