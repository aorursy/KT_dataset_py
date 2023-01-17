from keras.layers import Input, Reshape ,Dropout, Dense, BatchNormalization,ZeroPadding2D,Flatten,Activation

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D,Conv2D

from keras.models import Sequential,Model,load_model

from keras.optimizers import Adam

import numpy as np

from PIL import Image

from tqdm import tqdm

import os

# Generation resolution - Must be square 

# Training data is also scaled to this.

GENERATE_RES=2 #(1=32,2=64,....)

GENERATE_SQUARE=32*GENERATE_RES #will make up rows and columns of generated image...

IMAGE_CHANNELS=3



#PREVIEW IMAGE .....to see how our generator works...

PREVIEW_ROWS=4

PREVIEW_COLS=7

PREVIEW_MARGIN=16

SAVE_FREQ=100# for every 100 epochs....



# Size vector to generate images from

SEED_SIZE = 100



#configuration

DATA_PATH='../input/all-dogs'

EPOCHS = 10000

BATCH_SIZE = 32



print(f"Will generate {GENERATE_SQUARE}px square images.")

# Image set has 11,682 images.  Can take over an hour for initial preprocessing.

# Because of this time needed, save a Numpy preprocessed file.

# Note, that file is large enough to cause problems for sume verisons of Pickle,

# so Numpy binary files are used.



training_binary_path=os.path.join(DATA_PATH,f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')



print(f"Looking for file: {training_binary_path}")



if not os.path.isfile(training_binary_path):  #returns True if the file shown by path exists...

    print("loading training images")

    

    training_data=[]

    dogs_path=os.path.join(DATA_PATH,'all-dogs')

    for filename in tqdm(os.listdir(dogs_path)):

        path = os.path.join(dogs_path,filename)

        image=Image.open(path).resize((GENERATE_SQUARE,GENERATE_SQUARE),Image.ANTIALIAS)

        training_data.append(np.asarray(image))

    training_data = np.reshape(training_data,(-1,GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS))

    training_data = training_data / 127.5 - 1.

    

#    print("Saving training image binary...")      #good to save numpy file but here it dosent work because of read only system

  #  np.save(training_binary_path,training_data)



else:

    print(f"loading previous binary file")

    training_data=np.load(training_binary_path)
def build_generator(SEED_SIZE,channels):

    model=Sequential()

    

    model.add(Dense(4*4*256,activation='relu',input_dim=SEED_SIZE))

    model.add(Reshape((4,4,256)))

    

    model.add(UpSampling2D())

    model.add(Conv2D(256,kernel_size=3,padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation('relu'))

    

    model.add(UpSampling2D())

    model.add(Conv2D(256,kernel_size=3,padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation('relu'))

    

    # Output resolution, additional upsampling

    for i in range(GENERATE_RES):

        model.add(UpSampling2D())

        model.add(Conv2D(128,kernel_size=3,padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Activation('relu'))

        

    # Final CNN layer

    model.add(Conv2D(channels,kernel_size=3,padding="same"))

    model.add(Activation("tanh"))

    

    input = Input(shape=(SEED_SIZE,))

    generated_image = model(input)

    

    return Model(input,generated_image)





def build_discriminator(image_shape):

    model=Sequential()

    

    model.add(Conv2D(32,kernel_size=3,strides=2,input_shape=image_shape,padding='same'))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(64,kernel_size=3,strides=2,padding='same'))

    model.add(ZeroPadding2D(padding=((0,1),(0,1))))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))

    

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    

    

    input_image = Input(shape=image_shape)



    validity = model(input_image)

    return Model(input_image, validity)

    

    
image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)

optimizer = Adam(1.5e-4,0.5) # learning rate and momentum adjusted from paper



discriminator = build_discriminator(image_shape)

discriminator.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])



generator = build_generator(SEED_SIZE,IMAGE_CHANNELS)



random_input = Input(shape=(SEED_SIZE,))



generated_image = generator(random_input)



discriminator.trainable = False



validity = discriminator(generated_image)



combined = Model(random_input,validity)

combined.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])




y_real = np.ones((BATCH_SIZE,1))

y_fake = np.zeros((BATCH_SIZE,1))



fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))



cnt = 1

for epoch in range(EPOCHS):

    idx = np.random.randint(0,training_data.shape[0],BATCH_SIZE)

    x_real = training_data[idx]



    # Generate some images

    seed = np.random.normal(0,1,(BATCH_SIZE,SEED_SIZE))

    x_fake = generator.predict(seed)



    # Train discriminator on real and fake

    discriminator_metric_real = discriminator.train_on_batch(x_real,y_real)

    discriminator_metric_generated = discriminator.train_on_batch(x_fake,y_fake)

    discriminator_metric = 0.5 * np.add(discriminator_metric_real,discriminator_metric_generated)

    

    # Train generator on Calculate losses

    generator_metric = combined.train_on_batch(seed,y_real)

    

    print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")
seed = np.random.normal(0,1,(16,SEED_SIZE))

x_predict= generator.predict(seed)
import matplotlib.pyplot as plt
cnt=0

f,ax = plt.subplots(4,4) 

f.subplots_adjust(0,0,3,3)#(left,bottom,vertical_distance b/w columns,)

for i in range(0,4,1):

    for j in range(0,4,1):

        ax[i,j].imshow(x_predict[cnt])

        ax[i,j].axis('off')

        cnt=cnt+1
DATA_PATH_OUTPUT='../output_images/'

os.mkdir(DATA_PATH_OUTPUT)

import cv2



for i in range(10000):

    x_predict=generator.predict(np.random.normal(0,1,(1,SEED_SIZE)))

    path=os.path.join(DATA_PATH_OUTPUT,f'{i}.png')

    cv2.imwrite(path,x_predict)
import shutil

shutil.make_archive('images', 'zip', '../output_images')