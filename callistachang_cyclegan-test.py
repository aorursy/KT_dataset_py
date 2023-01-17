!pip install git+https://www.github.com/keras-team/keras-contrib.git
# example of using saved cyclegan models for image translation

from keras.models import load_model

from numpy import load

from numpy import vstack

from matplotlib import pyplot

from numpy.random import randint

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization



# load and prepare training images

def load_real_samples(filename):

	# load the dataset

	data = load(filename)

	# unpack arrays

	X1, X2 = data['arr_0'], data['arr_1']

	# scale from [0,255] to [-1,1]

	X1 = (X1 - 127.5) / 127.5

	X2 = (X2 - 127.5) / 127.5

	return [X1, X2]



# select a random sample of images from the dataset

def select_sample(dataset, n_samples):

	# choose random instances

	ix = randint(0, dataset.shape[0], n_samples)

	# retrieve selected images

	X = dataset[ix]

	return X



# plot the image, the translation, and the reconstruction

def show_plot(imagesX, imagesY1, imagesY2):

	images = vstack((imagesX, imagesY1, imagesY2))

	titles = ['Real', 'Generated', 'Reconstructed']

	# scale from [-1,1] to [0,1]

	images = (images + 1) / 2.0

	# plot images row by row

	for i in range(len(images)):

		# define subplot

		pyplot.subplot(1, len(images), 1 + i)

		# turn off axis

		pyplot.axis('off')

		# plot raw pixel data

		pyplot.imshow(images[i])

		# title

		pyplot.title(titles[i])

	pyplot.show()



# load dataset

A_data, B_data = load_real_samples('../input/horse2zebratest/horse2zebra_256.npz')

print('Loaded', A_data.shape, B_data.shape)

# load the models

cust = {'InstanceNormalization': InstanceNormalization}

model_AtoB = load_model('../input/horse2zebratest/g_model_AtoB_023740.h5', cust)

model_BtoA = load_model('../input/horse2zebratest/g_model_BtoA_023740.h5', cust)

# plot A->B->A

A_real = select_sample(A_data, 1)

B_generated  = model_AtoB.predict(A_real)

A_reconstructed = model_BtoA.predict(B_generated)

show_plot(A_real, B_generated, A_reconstructed)

# plot B->A->B

B_real = select_sample(B_data, 1)

A_generated  = model_BtoA.predict(B_real)

B_reconstructed = model_AtoB.predict(A_generated)

show_plot(B_real, A_generated, B_reconstructed)