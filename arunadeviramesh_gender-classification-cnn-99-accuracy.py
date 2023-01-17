import os

train_dir = '../input/genderdetectionface/dataset1/train'

validation_dir ='../input/genderdetectionface/dataset1/test'

# Directory with our training cat pictures

train_man_dir = os.path.join(train_dir, 'man')



# Directory with our training dog pictures

train_woman_dir = os.path.join(train_dir, 'woman')



# Directory with our validation cat pictures

validation_man_dir = os.path.join(validation_dir, 'man')



# Directory with our validation dog pictures

validation_woman_dir = os.path.join(validation_dir, 'woman')

train_cat_fnames = os.listdir(train_man_dir)

print(train_cat_fnames[:10])



train_dog_fnames = os.listdir(train_woman_dir)

train_dog_fnames.sort()

print(train_dog_fnames[:10])


%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

nrows = 4

ncols = 4

pic_index = 0
# Set up matplotlib fig, and size it to fit 4x4 pics

fig = plt.gcf()

fig.set_size_inches(ncols * 4, nrows * 4)



pic_index += 8

next_man_pix = [os.path.join(train_man_dir, fname) 

                for fname in train_cat_fnames[pic_index-8:pic_index]]

next_woman_pix = [os.path.join(train_woman_dir, fname) 

                for fname in train_dog_fnames[pic_index-8:pic_index]]



for i, img_path in enumerate(next_man_pix+next_woman_pix):

  # Set up subplot; subplot indices start at 1

  sp = plt.subplot(nrows, ncols, i + 1)

  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)

  plt.imshow(img)

plt.show()

from tensorflow.keras import layers

from tensorflow.keras import Model


# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for

# the three color channels: R, G, and B

img_input = layers.Input(shape=(150, 150, 3))



# First convolution extracts 16 filters that are 3x3

# Convolution is followed by max-pooling layer with a 2x2 window

x = layers.Conv2D(16, 3, activation='relu')(img_input)

x = layers.MaxPooling2D(2)(x)



# Second convolution extracts 32 filters that are 3x3

# Convolution is followed by max-pooling layer with a 2x2 window

x = layers.Conv2D(32, 3, activation='relu')(x)

x = layers.MaxPooling2D(2)(x)



# Third convolution extracts 64 filters that are 3x3

# Convolution is followed by max-pooling layer with a 2x2 window

x = layers.Conv2D(64, 3, activation='relu')(x)

x = layers.MaxPooling2D(2)(x)




# Flatten feature map to a 1-dim tensor so we can add fully connected layers

x = layers.Flatten()(x)



# Create a fully connected layer with ReLU activation and 512 hidden units

x = layers.Dense(512, activation='relu')(x)



# Create output layer with a single node and sigmoid activation

output = layers.Dense(1, activation='sigmoid')(x)



# Create model:

# input = input feature map

# output = input feature map + stacked convolution/maxpooling layers + fully 

# connected layer + sigmoid output layer

model = Model(img_input, output)
model.summary()
from tensorflow.keras.optimizers import RMSprop



model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=0.001),

              metrics=['acc'])




from tensorflow.keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)



# Flow training images in batches of 20 using train_datagen generator

train_generator = train_datagen.flow_from_directory(

        train_dir,  # This is the source directory for training images

        target_size=(150, 150),  # All images will be resized to 150x150

        batch_size=20,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



# Flow validation images in batches of 20 using val_datagen generator

validation_generator = val_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
history=model.fit(train_generator,

epochs = 15,

validation_data = validation_generator

)
%matplotlib inline



import matplotlib.image  as mpimg

import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, 'r', "Training Accuracy")

plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r', "Training Loss")

plt.plot(epochs, val_loss, 'b', "Validation Loss")

plt.figure()





# Desired output. Charts with training and validation metrics. No crash :)
# make a prediction for a new image.

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.models import load_model



# load and prepare the image

def load_image(filename):

	# load the image

	img = load_img(filename, target_size=(150,150))

	# convert to array

	img = img_to_array(img)

	# reshape into a single sample with 3 channels

	img = img.reshape(1,150,150, 3)

	# center pixel data

	img = img.astype('float32')

	img = img - [123.68, 116.779, 103.939]

	return img



# load an image and predict the class

	# load the image

img = load_image('../input/genderdetectionface/dataset1/valid/man/face_190.jpg')

# predict the class

result = model.predict(img)

if(result[0]==0):

    print("MAN")

else:

    print("WOMAN")
