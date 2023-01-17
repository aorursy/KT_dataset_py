%pylab inline
import cv2

from PIL import Image

import os
def is_correct_shape(img, shape):

    """ Return True if the image is the correct shape, False otherwise

        input:

            img - PIL image

            shape - 2-tuple containing (width, height) of the image

        output:

            True if correct shape, False if not

    """

    

    # Check if shape is correct

    if shape == img.size:

        return True

    else:

        return False

    

def open_check_img(dir_name, img_name, shape):

    """ Open an image and check if it is the correct shape. Return the image if

    correct, None otherwise

        input:

            dir_name - Name of the directory containing the image

            img_name - Filename for the image

            shape - 2-tuple containing (width, height) of the image

            ext - File extension DEFAULT .png

        output:

            Image if correct shape, None if not    

        """

    

    # Construct filepath

    filepath = dir_name + img_name

    

    # Open image file

    img = Image.open(filepath).convert('RGB')

    

    # Load img into memory, close image file

    img.load()



    # Check if image is the correct shape

    check = is_correct_shape(img, shape)

    

    # Return the image if it is the correct shape, otherwise return None

    if check:

        return img

    else:

        return None
# TODO: Make this concurrent so we can prepare a batch before it is used.

# Currently, loading images is the speed bottleneck; it takes longer for

# a batch to load than it does to train on a batch, and since only a

# single batch before each epoch, the model must wait to train on a

# new batch every iteration

class MyDatagen():

    def __init__(self, dir_name, correct_shape, batch_size):

        # Lazy solution to prevent problems, assume that of the 2135552 total images

        # at least 100000 are 28 x 28

        # Also serves to reduce training time by using a subset of all images

        self.hard_limit = 100000

        

        # Name of directory that contains images

        self.dir_name = dir_name

        

        # Correct image shape

        self.correct_shape = correct_shape

        

        # Number of images per batch

        self.batch_size = batch_size

        

        # Current image id to attempt loading

        self.current_index = 0

        

        # Number of images in the directory

        # TODO: This cuts off some legitimate images towards the highest index numbers, come up

        # with an elegant fix maybe

#         self.num_images = len(os.listdir(self.dir_name))

        

    def get_num_batches(self):

        """ Divide the total number of images by the batch size to find out how many batches

            are needed in total

            output:

                Number of batches to load per epoch"""

        

        # TODO: Find some way to confirm how many total 28 x 28 images are in the directory instead

        # of chopping off images at the end

        

        # Currently does not work since num_images is the total number of images, not number of

        # 28 x 28 images

#         return self.num_images // self.batch_size

        

    

        return self.hard_limit // self.batch_size

    

    def get_batch(self):

        """ Load a batch of images

            output:

                An np array of image np arrays of size batch_size"""

        # Count how many images have been loaded

        curr_loaded = 0

        

        # Contain images

        images = []

        

        # Try to load images until batch size is reached

        while curr_loaded < self.batch_size:

            try:

                # Check image

                img = open_check_img(self.dir_name, str(self.current_index).zfill(10) + 

                                     '.png', self.correct_shape)

                self.current_index += 1

            except:

                # Couldn't load image

                self.current_index += 1

                continue

                

            # Correct shape

            if img is not None:

                images.append(np.asarray(img))

                curr_loaded += 1

        

#         print('Finished loading batch of ' + str(curr_loaded) + ' images. Went through ' + 

#               str(self.current_index) + ' total IDs')

        

        # Reached batch size

        return np.asarray(images)
from keras.layers import Input, Dense, Flatten, Reshape, Dropout, BatchNormalization, Activation, ZeroPadding2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose

from keras.models import Sequential, Model

from keras.optimizers import Adam, SGD

from keras.utils import plot_model

import matplotlib.pyplot as plt
class DCGAN():

    def __init__(self):

        # Specify image size. We're using 28x28 for now

        self.image_rows = 28

        self.image_cols = 28

        # Our images are RGB and have 3 channels

        self.image_channels = 3

        # Put these together for the shape

        self.image_shape = (self.image_rows, self.image_cols, self.image_channels)

        # We will use latent dim to specify the size of the input noise vector

        self.latent_dim = 100

        

        # Optimizer

        optimizer = Adam(0.0002, 0.5)

        

        # Build the components of the GAN

        

        # Discriminator

        self.discriminator = self.create_discriminator()

        

        # Compile

        self.discriminator.compile(loss='binary_crossentropy',

                                  optimizer=Adam(0.0001, 0.5),

                                  metrics=['accuracy'])

        

        # Generator

        self.generator = self.create_generator()

        

        # Latent input vector

        z = Input(shape=(self.latent_dim,))

        img = self.generator(z)

        

        # Combined

        

        # The combined model only trains the generator

        self.discriminator.trainable = False

        

        # Discriminator takes generated images as input

        valid = self.discriminator(img)

        

        # Combined model consists of both discriminator and generator

        self.combined = Model(z, valid)

        self.combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))



    def create_generator(self):

        model = Sequential()

        

        # Number of nodes for conv layers

        c1 = 128

        c2 = 128

        c3 = 128

        c4 = 128

        

        # Convolution, Batch normalization, activation

        # Repeat

        

        # Initial layer, feed images into network

        model.add(Dense((256 * 7 * 7), input_dim=self.latent_dim))

        # Reshape to 7 x 7 x 256

        model.add(Reshape((7, 7, 256)))

        # Batch normalization

        model.add(BatchNormalization(momentum=0.9))

#         # Activation

#         model.add(LeakyReLU(alpha=0.1))

        # Convolution

        model.add(Conv2D(c1, kernel_size=5, padding='same'))

        # Batch normalization

        model.add(BatchNormalization(momentum=0.9))

        # Activation

        model.add(LeakyReLU(alpha=0.1))

        # Convolution Transpose from (7 x 7 x 128) to (14 x 14 x 128)

        model.add(Conv2DTranspose(c2, kernel_size=2, strides=4, padding='same'))

#         # Batch normalization

#         model.add(BatchNormalization(momentum=0.9))

        # Activation

        model.add(LeakyReLU(alpha=0.1))

#         # Convolution, transpose from (14 x 14 x 128) to (28 x 28 x 128)

#         model.add(Conv2DTranspose(c3, kernel_size=2, strides=2, padding='same'))

#         # Activation

#         model.add(LeakyReLU(alpha=0.1))

#         # Repeat        

        # Convolution

        model.add(Conv2D(c4, kernel_size=5, padding='same'))

        # Batch normalization

        model.add(BatchNormalization(momentum=0.9))

        # Activation

        model.add(LeakyReLU(alpha=0.1))

        # One more convolutional layer to force image to correct channels

        model.add(Conv2D(self.image_channels, kernel_size=5, padding='same'))

        # (28, 28, 3)

        # Limit pixels to (-1, 1)

        model.add(Activation("tanh"))

        

        # Information about the shape of the model and # of params

        model.summary()

        

        # Input noise

        noise = Input(shape=(self.latent_dim,))

        # Generated image

        img = model(noise)

        

        return Model(noise, img)

    

    def create_discriminator(self):

        model = Sequential()

        

        # Number of nodes for conv layers

        c1 = 128

        c2 = 128

        c3 = 128

        c4 = 128

        

        # Convolutional layer, batch normalization, activation, repeat

        model.add(Conv2D(c1, kernel_size=5, strides=2, padding='same', input_shape=self.image_shape))

        model.add(BatchNormalization(momentum=0.9))

        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(c2, kernel_size=5, strides=2, padding='same'))

        model.add(BatchNormalization(momentum=0.9))

        model.add(LeakyReLU(alpha=0.1))

        model.add(Conv2D(c3, kernel_size=5, strides=2, padding='same'))

        model.add(BatchNormalization(momentum=0.9))

        model.add(LeakyReLU(alpha=0.1))

#         model.add(Conv2D(c4, kernel_size=5, strides=2, padding='same'))

#         model.add(BatchNormalization(momentum=0.9))

#         model.add(LeakyReLU(alpha=0.1))

        # Dropout layer

        model.add(Dropout(0.4))

        # Flatten to 1D vector so we can prepare to simplify to a binary value

        model.add(Flatten())

        # Compress to a single binary value using a fully connected layer with sigmoid activation

        model.add(Dense(1, activation='sigmoid'))

        

        # Information about the shape of the model and # of params

        model.summary()

        

        # Takes a 28x28x3 image as input

        img = Input(shape=(self.image_shape))

        validity = model(img)

        

        return Model(img, validity)

    

    def train(self, epochs, batch_size=128, save_interval=50):

        # Ground truths

        valid = np.zeros((batch_size, 1))

        fake = np.ones((batch_size, 1))

        

        # Keep track of loss so it can be examined later

        self.g_loss_history = []

        self.d_loss_history = []

        self.d_loss_real_history = []

        self.d_loss_fake_history = []

        

        # Iterate and train model

        for epoch in range(epochs):

            # Create datagen object

            datagen = MyDatagen('../input/twitch_emotes/images/', (28, 28), batch_size)



            # Get total number of batches per epoch

            num_batches = datagen.get_num_batches()

            

            # Store sample images for experience replay in order to prevent mode collapse

            exp_replay = []

            

            # Train on a batch

            for i in range(num_batches):

                big_start = time.time()

                

                # Load batch

                start = time.time()

                train_x = datagen.get_batch()

                end = time.time()



                load_time = end - start



                # Rescale images between -1 and 1

                # Dividing by 127.5 scales between 0 and 2, subtracting 1 makes it -1 to 1

                train_x = train_x / 127.5 - 1



                # Random noise as input for generator

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                gen_images = self.generator.predict(noise)



                # Store a random image for experience replay

                r_idx = np.random.randint(batch_size)

                exp_replay.append(gen_images[r_idx])

                

                # Train discriminator

                

                # Once exp_replay has enough images, show them to the discriminator again

                if len(exp_replay) == batch_size:

                    replay_gen_images = np.array(exp_replay)

                    # Train discriminator on fake images with label smoothing

                    d_loss_fake = self.discriminator.train_on_batch(replay_gen_images, fake - (np.random.random((batch_size, 1)) * 0.1))

                    exp_replay = []

                else:

                    # Train discriminator on fake images with label smoothing

                    d_loss_fake = self.discriminator.train_on_batch(gen_images, fake - (np.random.random((batch_size, 1)) * 0.1))



                # Train discriminator on real images with label smoothing

                d_loss_real = self.discriminator.train_on_batch(train_x, valid  + (np.random.random((batch_size, 1)) * 0.1))



                # Average loss

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



                # Generator loss

                g_loss = self.combined.train_on_batch(noise, valid)



                if i % (num_batches // 20) == 0:

                    # Track progress, 5% of the way done with epoch

                    print ("[%d] batch: %d/%d [D loss: %f] [G loss: %f]" % (epoch, i, num_batches, d_loss[0], g_loss))

                    

                # Store incremental loss for later examination

                self.g_loss_history.append(g_loss)

                self.d_loss_history.append(d_loss[0])

                self.d_loss_real_history.append(d_loss_real[0])

                self.d_loss_fake_history.append(d_loss_fake[0])

                

                big_end = time.time()

                

                batch_time = big_end - big_start

                

#                 print('Training on batch took ' + str(batch_time) + ' seconds. Loading the batch took ' + str(load_time) + ' seconds. Loading took ' + str((load_time / batch_time) * 100) + '% of the total batch training time')

                

            if epoch % save_interval == 0:

                self.save_imgs(epoch)

            

    # Helper function for saving sample images during training.

    def save_imgs(self, epoch):

        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        gen_imgs = self.generator.predict(noise)

        

        # Rescale images 0 - 1

        gen_imgs = 0.5 * gen_imgs + 0.5

        

        fig, axs = plt.subplots(r, c)

        cnt = 0

        for i in range(r):

            for j in range(c):

                axs[i,j].imshow(gen_imgs[cnt])

                axs[i,j].axis('off')

                cnt += 1

        fig.savefig("emote_%d.png" % epoch)

        plt.close()
dcgan = DCGAN()



dcgan.train(epochs=8, batch_size=128, save_interval=1)
dcgan.discriminator.save('discriminator.h5')

dcgan.generator.save('generator.h5')
# Number of times to sample generator

num_samples = 10



# Show num_samples 4x4 grids of sample images

for i in range(num_samples):

    # Input for generator

    noise = np.random.normal(0, 1, (4 * 4, 100))



    # Use noise to generate images

    gen_imgs = dcgan.generator.predict(noise)



    # Rescale to [0, 1]

    gen_imgs = 0.5 * gen_imgs + 0.5



    # 4x4 figure of 16 images total

    fig, axs = plt.subplots(4, 4)

    cnt = 0

    for i in range(4):

        for j in range(4):

            axs[i,j].imshow(gen_imgs[cnt])

            axs[i,j].axis('off')

            cnt += 1
import matplotlib.pyplot as plt

# x axis values

x = np.arange(0., len(dcgan.d_loss_history), 1)

plt.plot(x, dcgan.d_loss_real_history, 'r-', x, dcgan.d_loss_fake_history, 'g-', x, dcgan.g_loss_history, 'b-', alpha=0.6, linewidth=0.5)

plt.legend(['D Loss Real', 'D Loss Fake', 'G Loss'], loc='upper right')

plt.title('GAN Loss Over Minibatches')

plt.xlabel('Minibatch')

plt.ylabel('Loss')