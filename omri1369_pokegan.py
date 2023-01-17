from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D

from tensorflow.keras.layers import UpSampling2D, Conv2D, LeakyReLU

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.optimizers import Adam

import numpy as np

from PIL import Image

from tqdm import tqdm

import os 

import matplotlib.pyplot as plt
# Run this for Google CoLab

from google.colab import drive

drive.mount('/content/drive')
# prepare data



main_path = '/content/drive/My Drive/pokemon_data/smd/'

training_paths = os.listdir(main_path)[1:]

# training_data = np.array([])



for i, im_path in tqdm(enumerate(training_paths)):

  im_path = main_path + im_path

  image = Image.open(im_path)

  # .resize((GENERATE_SQUARE,GENERATE_SQUARE),Image.ANTIALIAS)

  if i == 0:

    training_data = np.asarray(np.expand_dims(image, 0))

  else:

    training_data = np.vstack((training_data, np.expand_dims(image, 0)))



print(training_data.shape)



training_data_T = [np.fliplr(data) for data in training_data]

training_data_T = np.array(training_data_T)



train = np.concatenate((training_data, training_data_T))

np.save('/content/drive/My Drive/pokemon_data/PokiGan_training_data.npy', train)
training_data = np.load('/content/drive/My Drive/pokemon_data/PokiGan_training_data.npy')

training_data.shape
def build_generator(seed_size, channels):

    model = Sequential()



    model.add(Dense(4*4*256,activation="relu",input_dim=seed_size))

    model.add(Reshape((4,4,256)))



    model.add(UpSampling2D())

    model.add(Conv2D(256,kernel_size=3,padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))



    model.add(UpSampling2D())

    model.add(Conv2D(256,kernel_size=3,padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

   

    # Output resolution, additional upsampling

    for i in range(GENERATE_RES):

      model.add(UpSampling2D())

      model.add(Conv2D(128,kernel_size=3,padding="same"))

      model.add(BatchNormalization(momentum=0.8))

      model.add(Activation("relu"))



    # Final CNN layer

    model.add(Conv2D(channels,kernel_size=3,padding="same"))

    model.add(Activation("tanh"))



    input = Input(shape=(seed_size,))

    generated_image = model(input)



    return Model(input,generated_image)





def build_discriminator(image_shape):

    model = Sequential()



    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))

    model.add(LeakyReLU(alpha=0.2))



    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))

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

  

def save_images(cnt,noise):

  image_array = np.full(( 

      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 

      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 3), 

      255, dtype=np.uint8)

  

  generated_images = generator.predict(noise)



  generated_images = 0.5 * generated_images + 0.5



  image_count = 0

  for row in range(PREVIEW_ROWS):

      for col in range(PREVIEW_COLS):

        r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN

        c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN

        image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = generated_images[image_count] * 255

        image_count += 1



          

  output_path = os.path.join(DATA_PATH,'output')

  # output_path = DATA_PATH



  if not os.path.exists(output_path):

    os.makedirs(output_path)

  

  filename = os.path.join(output_path,f"train-{cnt}.png")

  im = Image.fromarray(image_array)

  im.save(filename)
GENERATE_RES = 2 # (1=32, 2=64, 3=96, etc.)

GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)

IMAGE_CHANNELS = 3



# Preview image 

PREVIEW_ROWS = 15

PREVIEW_COLS = 15

PREVIEW_MARGIN = 16

SAVE_FREQ_IMAGES = 100

SAVE_FREQ_MODEL = 500

# Size vector to generate images from

SEED_SIZE = 100



# Configuration

DATA_PATH = '/content/drive/My Drive/pokemon_data'

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

EPOCHS = 1000

BATCH_SIZE = 32



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



    # Time for an update?

    if epoch % SAVE_FREQ_IMAGES == 0:

        save_images(cnt, fixed_seed)

        cnt += 1

        print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")

    

    if epoch % SAVE_FREQ_MODEL == 0:

        generator.save(os.path.join(DATA_PATH,"PokiGenartor.h5"))

        print('genrator saved!')
fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))

save_images(cnt, fixed_seed)