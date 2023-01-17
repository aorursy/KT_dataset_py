#@title Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.
from __future__ import absolute_import, division, print_function, unicode_literals
# try:

#   # %tensorflow_version only exists in Colab.

#   %tensorflow_version 2.x

# except Exception:

#   pass

import tensorflow as tf
tf.__version__
# To generate GIFs

!pip install -q imageio
import glob

import imageio

import matplotlib.pyplot as plt

import numpy as np

import PIL

from tensorflow.keras import layers

import time, os, math, shutil

import pickle



import IPython

from IPython import display
BUFFER_SIZE = 60000
reduce_ = False

quarter = True

assert not (quarter and reduce_)

file_ending = ("_quarter_64" if quarter else "") + ("_reduced" if reduce_ else "")

side_length = 32 if reduce_ else 64
# Load data (each row corresponds to one sample)

x_train = np.loadtxt("../input/quarter/x_train{}.csv".format(file_ending), dtype=np.float64, delimiter=',')

x_test  = np.loadtxt("../input/quarter/x_test{}.csv".format(file_ending), dtype=np.float64, delimiter=',')



# Reshape x to recover its 2D content

x_train = x_train.reshape(x_train.shape[0], side_length, side_length, 1)

x_test = x_test.reshape(x_test.shape[0], side_length, side_length, 1)

print(x_train.shape)

print(x_test.shape)
# Load labels:

y_train = np.loadtxt("../input/quarter/y_test.csv", dtype=np.float64, delimiter=',')

y_test = np.loadtxt("../input/quarter/y_test.csv", dtype=np.float64, delimiter=',')



# Transform the labels y so that min(y) == 0 and max(y) == 1. Importantly, y_train and y_test must be considered jointly.

def up_scale(*args):

    """ Up-scale features from ~10^-9 to ~0

    """

    for x in args:

        yield x * 10**9

        

y_train, y_test = up_scale(y_train, y_test)

#y_train, y_test = standardize_zero_one(y_train, y_test)

print("Mean and standard deviation of y_train", np.mean(y_train), np.std(y_train))

print("Mean and standard deviation of y_test", np.mean(y_test), np.std(y_test))
x = np.concatenate((x_train, x_test), axis=0)

y = np.concatenate((y_train, y_test), axis=0)
BATCH_SIZE = 128



# Batch and shuffle the data

dataset = tf.data.Dataset.from_tensor_slices(x).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
def make_generator_model(noise_dim):

    model = tf.keras.Sequential()

    model.add(layers.Dense(side_length*side_length*16, use_bias=False, input_shape=(noise_dim,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Reshape((side_length//4, side_length//4, 256)))

    assert model.output_shape == (None, side_length//4, side_length//4, 256) # Note: None is the batch size



    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    assert model.output_shape == (None, side_length//4, side_length//4, 128)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, side_length//2, side_length//2, 64)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))

    assert model.output_shape == (None, side_length, side_length, 1)



    return model
def make_discriminator_model():

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',

                                     input_shape=[side_length, side_length, 1]))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Flatten())

    model.add(layers.Dense(1))



    return model
# load json and create model

with open('../input/psi2-predictor-191212/model.json', 'r') as json_file:

    loaded_model_json = json_file.read()

psi2_model = tf.keras.models.model_from_json(loaded_model_json)

# load weights into new model

psi2_model.load_weights("../input/psi2-predictor-191212/model.h5")

for l in psi2_model.layers:

    l.trainable = False

psi2_model.compile(loss='mean_squared_error')

print("Loaded model from disk")
print(psi2_model.summary())
# This method returns a helper function to compute cross entropy loss

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss
def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)
class Images:

    """This class defines only class variables and class methods. It needn't be initiated.

    

    It was created to tackle the file generation limit of 500 files on kaggle. Unfortunately,

    it seems that the kaggle kernel does not keep track of how many files exist, but rather

    counts how many files were created (regardless of later destruction).

    """

    zipzip_count = 0

    zip_count = 0

    image_count = 0

    zip_dir = "/kaggle/working/archives"

    MAX_ZIP_COUNT = 70

    

    def cleanup():

        """Create a zip archive with all of the figures in TestRun.base_dir, and delete the figures.

        """

        zip_name = "output_figures_{}".format(Images.zip_count + Images.MAX_ZIP_COUNT * Images.zipzip_count)

        shutil.make_archive(os.path.join(Images.zip_dir, zip_name), 'zip', TestRun.base_dir)

#         !zip -r {os.path.join(Images.zip_dir, zip_name)} {TestRun.base_dir}/

        !rm -rf  {TestRun.base_dir}/*

    

    def archive_zips():

        """Create a zip archive with all of the zip archives created by the Images.cleanup() function,

        and subsequently delete them.

        """

        zipzip_name = "/kaggle/working/output_archives_{}".format(Images.zipzip_count)

        shutil.make_archive(zipzip_name, 'zip', Images.zip_dir)

#         !zip -r {zipzip_name} {Images.zip_dir}/

        !rm -rf  {Images.zip_dir}/*

        if not os.path.isdir(Images.zip_dir):

            os.makedirs(Images.zip_dir)

        Images.zipzip_count += 1

        Images.zip_count = 0

        

    def prepare(target_dir):

        """This function is called whenever a figure will be created.

        It keeps track of the current number of figures in the folder, and ensures that

        the target directory exists.

        """

        Images.increment_image_count()

        if not os.path.isdir(target_dir):

            os.makedirs(target_dir)

        

    def increment_image_count():

        """Increment the image counter. If it exceeds the specified limit,

        create a zip archive of the figures and subsequently remove them.

        """

        Images.image_count += 1

        if Images.image_count + Images.zip_count + Images.zipzip_count > 400 :

            if Images.zip_count > MAX_ZIP_COUNT :

                Images.archive_zips()

            else:

                Images.cleanup()

                Images.zip_count += 1

    

    def generate_and_save(model, epoch, test_input, figure_dir):

        """Use the given model to generate 16 invisibility cloaks and plot them in a figure.

        """

        Images.prepare(figure_dir)

        # Notice `training` is set to False.

        # This is so all layers run in inference mode (batchnorm).

        predictions = model(test_input, training=False)



        fig = plt.figure(figsize=(4,4))



        for i in range(predictions.shape[0]):

            plt.subplot(4, 4, i+1)

            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')

            plt.axis('off')



        plt.savefig(os.path.join(figure_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))

#         plt.show()

        

    def make_gif(figure_dir):

        """Use the images in the directory figure_dir to generate a gif

        showing how the generator improves with each epoch.

        """

        Images.prepare(figure_dir)

        anim_file = os.path.join(figure_dir, 'dcgan.gif')



        with imageio.get_writer(anim_file, mode='I') as writer:

            filenames = glob.glob(os.path.join(figure_dir, 'image*.png'))

            filenames = sorted(filenames)

            last = -1

            for i,filename in enumerate(filenames):

                frame = 2*(i**0.5)

                if round(frame) > round(last):

                    last = frame

                else:

                    continue

                image = imageio.imread(filename)

                writer.append_data(image)

            image = imageio.imread(filename)

            writer.append_data(image)

#         if IPython.version_info > (6,2,0,''):

#             display.Image(filename=anim_file)

            

    def plot_and_save(lossRecorder, figure_dir):

        """Plot the evolution of generator loss, discriminator loss, and Psi2,

        at each epoch of one training loop.

        Takes as input a LossRecorder instance.

        """

        Images.prepare(figure_dir)

        

        fig = plt.figure(figsize=(4,4))

        plt.plot(lossRecorder.psi2_losses, linestyle='--')

        ax = plt.gca()

        ax.set_ylabel("Psi2")

        ax.legend(["Psi2"])

        ax = ax.twinx()

        ax.plot(lossRecorder.gen_losses)

        ax.plot(lossRecorder.disc_losses)

        plt.legend(["Generator loss", "Discriminator loss"])

        plt.xlabel("Epoch")

        ax.set_ylabel("Cross-entropy")

        plt.title("Evolution of the GAN")

        plt.savefig(os.path.join(figure_dir, 'Losses_vs_epochs'))

#         plt.show()
class LossRecorder:

    def __init__(self):

        self.psi2_losses = []

        self.gen_losses = []

        self.disc_losses = []

    

    def append(self, losses):

        self.psi2_losses.append(losses[0])

        self.gen_losses.append(losses[1])

        self.disc_losses.append(losses[2])

        

    def append_mean_of(self, loss_recorder):

        psi2_loss, gen_loss, disc_loss = loss_recorder.mean()

        self.psi2_losses.append(psi2_loss)

        self.gen_losses.append(gen_loss)

        self.disc_losses.append(disc_loss)

    

    def mean(self):

        return np.mean(self.psi2_losses), np.mean(self.gen_losses), np.mean(self.disc_losses)

    

    def last(self):

        return self.psi2_losses[-1], self.gen_losses[-1], self.disc_losses[-1]

    
class UninitializedException(Exception):

    def __init__(self):

        self.message = "TestRun must always be reinitialized before training!"
class TestRun:

    """Generate and train the models of the GAN for a given set of parameters"""

    num_examples_to_generate = 16 # Number of images to generate for the gif.

    base_dir = '/kaggle/working/figures'

    

    def __init__(self, noise_dim):

        """A new TestRun instance must be created for each change in noise_dim.

        The same seed is then used to generate 16 invisibility cloaks (for show) for all psi2_factors.

        """

        self.noise_dim = noise_dim

        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])

        self.initialized = False

    

    def reinitialize(self, psi2_scale_factor):

        """Each time the psi2_scale_factor is changed, the TestRun class must be reinitialized.

        Configure the path where images are saved,

        create a new generator and a new discriminator,

        and create new optimizers for the two models.

        """

        self.initialized = True;

        self.psi2_scale_factor = psi2_scale_factor

        

        self.figure_dir = os.path.join(self.base_dir, "noise_dim_{}".format(self.noise_dim), "psi2_factor_{}".format(self.psi2_scale_factor))

        

        self.generator = make_generator_model(self.noise_dim)

        self.discriminator = make_discriminator_model()

        

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    

    def train_step(self, images):

        """Perform one training step (one batch of images).

        Start by generating a random noise vector.

        Then, generate images and compute the losses.

        Use tf.GradientTape to compute and apply gradients.

        Return the losses.

        """

        noise = tf.random.normal([BATCH_SIZE, self.noise_dim])



        with tf.GradientTape(watch_accessed_variables=False) as gen_tape, tf.GradientTape(watch_accessed_variables=False) as disc_tape:

            gen_tape.watch(self.generator.trainable_variables)

            disc_tape.watch(self.discriminator.trainable_variables)



            generated_images = self.generator(noise, training=True)



            real_output = self.discriminator(images, training=True)

            fake_output = self.discriminator(generated_images, training=True)



            psi2_loss = np.mean(psi2_model.predict(generated_images))

            gen_loss = generator_loss(fake_output)

            disc_loss = discriminator_loss(real_output, fake_output)

#             print("Generator loss:", gen_loss)

#             print("Psi2 loss:", psi2_loss.shape)



            total_generator_loss = (gen_loss + self.psi2_scale_factor * psi2_loss) / (1 + self.psi2_scale_factor)



        gradients_of_generator = gen_tape.gradient(total_generator_loss, self.generator.trainable_variables)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)



        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))



        return psi2_loss, gen_loss, disc_loss

    

    def train(self, dataset, epochs):

        """Train the GAN for the given number of epochs on the given dataset.

        Loop over the epochs and train batchwise.

        """

        if not self.initialized:

            raise UninitializedException()

        self.initialized = False

        

        losses_by_epoch = LossRecorder()

        for epoch in range(epochs):

            start = time.time()



            losses_by_image = LossRecorder()

            for image_batch in dataset:

                losses_by_image.append(

                    self.train_step(image_batch)

                )

            losses_by_epoch.append_mean_of(losses_by_image)



            # We can't produce beautiful gifs because kaggle won't let us create more than 500 files even if we regularly archive & delete them :(

            # Produce images for the GIF as we go

    #         display.clear_output(wait=True)

#             Images.generate_and_save(self.generator,

#                                      epoch + 1,

#                                      self.seed,

#                                      self.figure_dir)

#             # Save the model every 15 epochs

#             if (epoch + 1) % 15 == 0:

#                 checkpoint.save(file_prefix = checkpoint_prefix)



            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



        # Generate after the final epoch

#         display.clear_output(wait=True)

        Images.generate_and_save(self.generator,

                                 epochs,

                                 self.seed,

                                 self.figure_dir)

        Images.plot_and_save(losses_by_epoch, self.figure_dir)

#         Images.make_gif(self.figure_dir)

        

        return losses_by_epoch.last()

        
class DcganGridsearch:

    """Find the best parameters noise_dim and psi2_scale_factor

    """

    pickle_filepath = '/kaggle/working/grid_search_results.pickle'

    MAX_RUN_TIME = 28800 # Kernel is killed after 32400s (9h) -> 28800 (8h) leaves 1h

    

    def __init__(self, epochs, noise_dims, psi2_factors):

        self.start_time = time.time()

        self.cleanup_done = False

        

        self.epochs = epochs

        self.noise_dims = noise_dims

        self.psi2_factors = psi2_factors

        

        self.grid_shape = (len(noise_dims), len(psi2_factors))

        self.psi2_losses = np.zeros(self.grid_shape)

        self.gen_losses = np.zeros(self.grid_shape)

        self.disc_losses = np.zeros(self.grid_shape)

    

    def search(self):

        """Perform a grid search (two nested for loops).

        Return the generator (for generating and showing images)."""

        for i, noise_dim in enumerate(self.noise_dims):

            test_run = TestRun(noise_dim)



            for j, psi2_scale_factor in enumerate(self.psi2_factors):

                if self.check_remaining_time():

                    break

                

                test_run.reinitialize(psi2_scale_factor)

                psi2_loss, gen_loss, disc_loss = test_run.train(dataset, self.epochs)

                self.psi2_losses[i,j] = psi2_loss

                self.gen_losses[i,j] = gen_loss

                self.disc_losses[i,j] = disc_loss

        generator = test_run.generator

        del test_run

        return generator

    

    def pickle(self):

        """Serialize the grid search results"""

        pickle_dict = {

            'noise_dims': self.noise_dims,

            'psi2_factors': self.psi2_factors,

            'psi2_losses': self.psi2_losses,

            'gen_losses': self.gen_losses,

            'disc_losses': self.disc_losses,

        }

        with open(self.pickle_filepath, 'wb') as dumpfile:

            pickle.dump(pickle_dict, dumpfile)

    

    def check_remaining_time(self):

        if time.time() > self.start_time + self.MAX_RUN_TIME :

            self.cleanup()

            return True

        return False

    

    def cleanup(self):

        if not self.cleanup_done:

            Images.cleanup()

            Images.archive_zips()

            self.cleanup_done = True

    

    def get_best_psi2(self):

        """Zip and remove figures etc."""

        best_psi2_by_factor = self.psi2_losses.min(axis=0)

        assert len(best_psi2_by_factor.shape) == 1

        assert best_psi2_by_factor.shape[0] == len(self.psi2_factors)

        best_factor_ind = np.argmin(best_psi2_by_factor)

        best_noisedims_ind = np.argmin(self.psi2_losses[:,best_factor])

        best_psi2 = self.psi2_losses[best_noisedims_ind, best_factor_ind]

        assert np.isclose(best_psi2, self.psi2_losses.min())

        return best_psi2, self.noise_dims[best_noisedims_ind], self.psi2_factors[best_factor_ind]

        

        
#if you get error messages, make sure you have the latest version of tensorflow (then it should work, not tested though)

#if still not working please fork and run the latest version published on kaggle that you can find here:

# https://www.kaggle.com/froehlichbergbier/dc-gan-for-invisibility-cloak



dcgan_gs = DcganGridsearch(

    epochs= 40,

    noise_dims= [64], # np.power(2, range(4, 10)),

    psi2_factors= [2.5] # [0.1, 0.8, 2.5, 10]

)



last_generator = dcgan_gs.search()

dcgan_gs.pickle()

dcgan_gs.cleanup()
# Generate the invisibility cloaks (without running out of memory...)

NUM_BATCHES = 50

BATCH_SIZE = 1000

psi2_of_images = []

seed = tf.random.normal([BATCH_SIZE*NUM_BATCHES, dcgan_gs.noise_dims[-1]])

for batch in range(NUM_BATCHES):

    generated_images = last_generator(seed[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE], training=False)

    psi2_of_images = np.concatenate((psi2_of_images, psi2_model.predict(generated_images).flat), axis=0)

    del generated_images

assert len(psi2_of_images.shape) == 1

assert psi2_of_images.shape[0] == NUM_BATCHES * BATCH_SIZE



# Visualize the predicted Psi2 values

plt.boxplot(psi2_of_images)

plt.ylabel("Psi 2")

plt.title("Box-plot of the generated images' Psi 2 metric")

plt.savefig(os.path.join(TestRun.base_dir, "generator_box_plot"))



# Find the best invisibility cloaks (after we had to delete them...) and plot them

best_indices = np.argsort(psi2_of_images)[:16]



best_images = np.concatenate([last_generator(seed[ind:ind+1,:], training=False) for ind in best_indices], axis=0)



fig = plt.figure(figsize=(8,2))



for i in range(16):

    plt.subplot(2,8, i+1)

    plt.imshow(best_images[i,:, :, 0] * 127.5 + 127.5, cmap='gray')

    plt.axis('off')



plt.savefig(os.path.join(TestRun.base_dir, "16 best images"))



# Save the best invisibility cloaks to CSV format

np.savetxt("/kaggle/working/best16images.csv", best_images.reshape((16*64, 64)), delimiter=',', fmt='%i')



print(psi2_of_images[best_indices])