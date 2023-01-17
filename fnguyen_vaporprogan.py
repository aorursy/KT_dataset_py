# example of defining composite models for the progressive growing gan

from numpy import expand_dims

from numpy import zeros

from numpy import ones

from numpy import vstack

from numpy.random import randn

from numpy.random import randint

from keras.optimizers import Adam

from keras.models import Sequential

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Activation

from keras.layers import Reshape

from keras.layers import Conv2D

from keras.layers import UpSampling2D

from keras.layers import AveragePooling2D

from keras.layers import LeakyReLU

from keras.layers import BatchNormalization

from keras.layers import Add

from keras.utils.vis_utils import plot_model

from keras import backend

from skimage.transform import resize

import matplotlib.pyplot as pyplot

import sys





# weighted sum output

class WeightedSum(Add):

	# init with default value

	def __init__(self, alpha=0.0, **kwargs):

		super(WeightedSum, self).__init__(**kwargs)

		self.alpha = backend.variable(alpha, name='ws_alpha')



	# output a weighted sum of inputs

	def _merge_function(self, inputs):

		# only supports a weighted sum of two inputs

		assert (len(inputs) == 2)

		# ((1-a) * input1) + (a * input2)

		output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])

		return output



def load_real_samples():

	# load Vaporarray dataset

	X = np.load('../input/vaporarray/test.out.npy')

	return X



def generate_real_samples(dataset, n_samples):

	# choose random instances

	ix = randint(0, dataset.shape[0], n_samples)

	# retrieve selected images

	X = dataset[ix]

	# generate 'real' class labels (1)

	y = np.random.uniform(low=0.8, high=1, size=(n_samples,1))

	return X, y



# generate points in latent space as input for the generator

def generate_latent_points(latent_dim, n_samples):

	# generate points in the latent space

	x_input = randn(latent_dim * n_samples)

	# reshape into a batch of inputs for the network

	x_input = x_input.reshape(n_samples, latent_dim)

	return x_input



# use the generator to generate n fake examples, with class labels

def generate_fake_samples(g_model, latent_dim, n_samples):

	# generate points in latent space

	x_input = generate_latent_points(latent_dim, n_samples)

	# predict outputs

	X = g_model.predict(x_input)

	# create 'fake' class labels (0)

	y = zeros((n_samples, 1))

	return X, y



# create and save a plot of generated images

def save_plot(examples, epoch, n=7):

	# plot images

	examples

	for i in range(n * n):

		# define subplot

		pyplot.subplot(n, n, 1 + i)

		# turn off axis

		pyplot.axis('off')

		# plot raw pixel data

		pyplot.imshow(examples[i])

	# save plot to file

	filename = 'vaporwave_e%03d.png' % (epoch+1)

	pyplot.savefig(filename)

	pyplot.close()

    

# evaluate the discriminator, plot generated images, save generator model

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):

	# prepare fake examples

	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)

	# save plot

	save_plot(x_fake, epoch)

	# save the generator model tile file

	filename = 'generator_model_%03d.h5' % (epoch+1)

	g_model.save(filename)



# add a discriminator block

def add_discriminator_block(old_model, n_input_layers=3):

	# get shape of existing model

	in_shape = list(old_model.input.shape)

	# define new input shape as double the size

	input_shape = (in_shape[-2].value*2, in_shape[-2].value*2, in_shape[-1].value)

	in_image = Input(shape=input_shape)

	# define new input processing layer

	d = Conv2D(64, (1,1), padding='same', kernel_initializer='he_normal')(in_image)

	d = LeakyReLU(alpha=0.2)(d)

	# define new block

	d = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(d)

	d = BatchNormalization()(d)

	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(d)

	d = BatchNormalization()(d)

	d = LeakyReLU(alpha=0.2)(d)

	d = AveragePooling2D()(d)

	block_new = d

	# skip the input, 1x1 and activation for the old model

	for i in range(n_input_layers, len(old_model.layers)):

		d = old_model.layers[i](d)

	# define straight-through model

	model1 = Model(in_image, d)

	# compile model

	model1.compile(loss='mse', optimizer=Adam(lr=0.0004, beta_1=0, beta_2=0.99, epsilon=10e-8))

	# downsample the new larger image

	downsample = AveragePooling2D()(in_image)

	# connect old input processing to downsampled new input

	block_old = old_model.layers[1](downsample)

	block_old = old_model.layers[2](block_old)

	# fade in output of old model input layer with new input

	d = WeightedSum()([block_old, block_new])

	# skip the input, 1x1 and activation for the old model

	for i in range(n_input_layers, len(old_model.layers)):

		d = old_model.layers[i](d)

	# define straight-through model

	model2 = Model(in_image, d)

	# compile model

	model2.compile(loss='mse', optimizer=Adam(lr=0.0004, beta_1=0, beta_2=0.99, epsilon=10e-8))

	return [model1, model2]



# define the discriminator models for each image resolution

def define_discriminator(n_blocks, input_shape=(4,4,3)):

	model_list = list()

	# base model input

	in_image = Input(shape=input_shape)

	# conv 1x1

	d = Conv2D(64, (1,1), padding='same', kernel_initializer='he_normal')(in_image)

	d = LeakyReLU(alpha=0.2)(d)

	# conv 3x3 (output block)

	d = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(d)

	d = BatchNormalization()(d)

	d = LeakyReLU(alpha=0.2)(d)

	# conv 4x4

	d = Conv2D(128, (4,4), padding='same', kernel_initializer='he_normal')(d)

	d = BatchNormalization()(d)

	d = LeakyReLU(alpha=0.2)(d)

	# dense output layer

	d = Flatten()(d)

	out_class = Dense(1)(d)

	# define model

	model = Model(in_image, out_class)

	# compile model

	model.compile(loss='mse', optimizer=Adam(lr=0.0004, beta_1=0, beta_2=0.99, epsilon=10e-8))

	# store model

	model_list.append([model, model])

	# create submodels

	for i in range(1, n_blocks):

		# get prior model without the fade-on

		old_model = model_list[i - 1][0]

		# create new model for next resolution

		models = add_discriminator_block(old_model)

		# store model

		model_list.append(models)

	return model_list



# add a generator block

def add_generator_block(old_model):

	# get the end of the last block

	block_end = old_model.layers[-2].output

	# upsample, and define new block

	upsampling = UpSampling2D()(block_end)

	g = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(upsampling)

	g = BatchNormalization()(g)

	g = LeakyReLU(alpha=0.2)(g)

	g = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(g)

	g = BatchNormalization()(g)

	g = LeakyReLU(alpha=0.2)(g)

	# add new output layer

	out_image = Conv2D(3, (1,1), padding='same', kernel_initializer='he_normal')(g)

	# define model

	model1 = Model(old_model.input, out_image)

	# get the output layer from old model

	out_old = old_model.layers[-1]

	# connect the upsampling to the old output layer

	out_image2 = out_old(upsampling)

	# define new output image as the weighted sum of the old and new models

	merged = WeightedSum()([out_image2, out_image])

	# define model

	model2 = Model(old_model.input, merged)

	return [model1, model2]



# define generator models

def define_generator(latent_dim, n_blocks, in_dim=4):

	model_list = list()

	# base model latent input

	in_latent = Input(shape=(latent_dim,))

	# linear scale up to activation maps

	g  = Dense(128 * in_dim * in_dim, kernel_initializer='he_normal')(in_latent)

	g = Reshape((in_dim, in_dim, 128))(g)

	# conv 4x4, input block

	g = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(g)

	g = BatchNormalization()(g)

	g = LeakyReLU(alpha=0.2)(g)

	# conv 3x3

	g = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(g)

	g = BatchNormalization()(g)

	g = LeakyReLU(alpha=0.2)(g)

	# conv 1x1, output block

	out_image = Conv2D(3, (1,1), padding='same', kernel_initializer='he_normal', activation = 'sigmoid')(g)

	# define model

	model = Model(in_latent, out_image)

	# store model

	model_list.append([model, model])

	# create submodels

	for i in range(1, n_blocks):

		# get prior model without the fade-on

		old_model = model_list[i - 1][0]

		# create new model for next resolution

		models = add_generator_block(old_model)

		# store model

		model_list.append(models)

	return model_list



# define composite models for training generators via discriminators

def define_composite(discriminators, generators):

	model_list = list()

	# create composite models

	for i in range(len(discriminators)):

		g_models, d_models = generators[i], discriminators[i]

		# straight-through model

		d_models[0].trainable = False

		model1 = Sequential()

		model1.add(g_models[0])

		model1.add(d_models[0])

		model1.compile(loss='mse', optimizer=Adam(lr=0.0001, beta_1=0, beta_2=0.99, epsilon=10e-8))

		# fade-in model

		d_models[1].trainable = False

		model2 = Sequential()

		model2.add(g_models[1])

		model2.add(d_models[1])

		model2.compile(loss='mse', optimizer=Adam(lr=0.0001, beta_1=0, beta_2=0.99, epsilon=10e-8))

		# store

		model_list.append([model1, model2])

	return model_list



# define models

discriminators = define_discriminator(6)

# define models

generators = define_generator(100, 6)

# define composite models

composite = define_composite(discriminators, generators)
# update the alpha value on each instance of WeightedSum

def update_fadein(models, step, n_steps):

	# calculate current alpha (linear from 0 to 1)

	alpha = step / float(n_steps - 1)

	# update the alpha for each model

	for model in models:

		for layer in model.layers:

			if isinstance(layer, WeightedSum):

				backend.set_value(layer.alpha, alpha)

                

# train a generator and discriminator

def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):

	# calculate the number of batches per training epoch

	bat_per_epo = int(dataset.shape[0] / n_batch)

	# calculate the number of training iterations

	n_steps = bat_per_epo * n_epochs

	# calculate the size of half a batch of samples

	half_batch = int(n_batch / 2)

	# manually enumerate epochs

	for i in range(n_steps):

		# update alpha for all WeightedSum layers when fading in new blocks

		if fadein:

			update_fadein([g_model, d_model, gan_model], i, n_steps)

		# prepare real and fake samples

		X_real, y_real = generate_real_samples(dataset, half_batch)

		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

		# update discriminator model

		d_loss1 = d_model.train_on_batch(X_real, y_real)

		d_loss2 = d_model.train_on_batch(X_fake, y_fake)

		# update the generator via the discriminator's error

		z_input = generate_latent_points(latent_dim, n_batch)

		y_real2 = ones((n_batch, 1))

		g_loss = gan_model.train_on_batch(z_input, y_real2)

        

# scale images to preferred size

def scale_dataset(images, new_shape):

	images_list = list()

	for image in images:

		# resize with nearest neighbor interpolation

		new_image = resize(image, new_shape, 0)

		# store

		images_list.append(new_image)

	return np.asarray(images_list)



# train the generator and discriminator

def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):

	# fit the baseline model

	g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]

	# scale dataset to appropriate size

	gen_shape = g_normal.output_shape

	scaled_data = scale_dataset(dataset, gen_shape[1:])

	print('Scaled Data', scaled_data.shape)

	# train normal or straight-through models

	train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm, n_batch)

	# process each level of growth

	for i in range(1, len(g_models)):

		# retrieve models for this level of growth

		[g_normal, g_fadein] = g_models[i]

		[d_normal, d_fadein] = d_models[i]

		[gan_normal, gan_fadein] = gan_models[i]

		# scale dataset to appropriate size

		gen_shape = g_normal.output_shape

		scaled_data = scale_dataset(dataset, gen_shape[1:])

		print('Scaled Data', scaled_data.shape)

		# train fade-in models for next level of growth

		train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein, n_batch, True)

		# train normal or straight-through models

		train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm, n_batch)

		summarize_performance(i, g_normal, d_normal, dataset, latent_dim)
import numpy as np

# number of growth phase, e.g. 3 = 16x16 images, 6 = 128x128 images

n_blocks = 7

# size of the latent space

latent_dim = 100

# define models

d_models = define_discriminator(n_blocks)

# define models

g_models = define_generator(100, n_blocks)

# define composite models

gan_models = define_composite(d_models, g_models)

# load image data

dataset = load_real_samples()

# train model

train(g_models, d_models, gan_models, dataset, latent_dim, 200, 200, 8)