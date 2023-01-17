# Import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import os

import pickle as pkl



import xml.etree.ElementTree as ET 

from PIL import Image 



from keras.preprocessing.image import load_img



print(os.listdir("../input"))
# List of images

images_folder = "../input/all-dogs/all-dogs"

annotation_folder = r'../input/annotation/Annotation'

filenames = os.listdir(images_folder)

breeds = os.listdir(annotation_folder)
# Load images as array (64x64x3)

# Credit goes to : https://www.kaggle.com/paulorzp/show-annotations-and-breeds

crop_dogspic = True

idx = 0

names = []

img_height, img_width = 64, 64

images = np.zeros((len(filenames), img_height, img_width, 3)) # No of samples x image_height x image_width x no of channels





for breed in breeds:

	path = os.path.join(annotation_folder, breed)

	for dog in os.listdir(path):

		if crop_dogspic:

			# Load image

			try: 

				image_path = os.path.join(images_folder, dog + r".jpg")

				img = Image.open(image_path)



				# Extract the bounding box 

				annotation_path = os.path.join(annotation_folder, breed, dog)

				tree = ET.parse(annotation_path)

				objects = tree.getroot().findall("object")



				for obj in objects:

					# For each object, get the bounding box coordinates

					bndbox = obj.find("bndbox")

					xmin = int(bndbox.find("xmin").text)

					ymin = int(bndbox.find("ymin").text)

					xmax = int(bndbox.find("xmax").text)

					ymax = int(bndbox.find("ymax").text)



					# Calculate the minimum difference for cropping

					min_diff = np.min((xmax - xmin, ymax - ymin))



					# Crop image

					img_cropped = img.crop((xmin, ymin, xmin + min_diff, ymin + min_diff))

					img_cropped = img_cropped.resize((img_height, img_width))



					# Save details as array

					images[idx, :, :, :] = np.asarray(img_cropped) # Images as array of number

					names.append(breed) # Category (breed in this case)



					idx += 1

			except: 

				pass

		else:

			try:

				image_path = os.path.join(images_folder, dog + r".jpg")

				img = Image.open(image_path)

				img = img.resize((img_height, img_width))



				# Save to array

				images[idx, :, :, :] = np.asarray(img)

				names.append(breed)

				idx += 1

			except:

				pass
# Shuffle the dataset

idx = np.arange(idx)

np.random.shuffle(idx)

images = images[idx, :, :, :]

names = np.array(names)[idx]
# Display images

sample_idx = np.random.randint(0, len(idx), 25)

fig, axes = plt.subplots(5, 5, figsize = (12, 12))

for ii, ax in zip(sample_idx, axes.flatten()):

	img = Image.fromarray(images[ii, :, :, :].astype("uint8"))

	ax.imshow(img)

	ax.xaxis.set_visible(False)

	ax.yaxis.set_visible(False)

	ax.title.set_text(names[ii].split("-")[1])
# Network inputs

def model_inputs(real_dim, z_dim):

	inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name = "input_real")

	inputs_z = tf.placeholder(tf.float32, (None, z_dim), name = "input_z")



	return inputs_real, inputs_z
# Generator network

def generator(z, output_dim, reuse = False, alpha = 0.2, training = True, initializer = tf.random_normal_initializer(0, 0.02)):

	with tf.variable_scope("generator", reuse = reuse):

		# First fully connected layer

		x1 = tf.layers.dense(inputs = z, units = 4*4*1024, kernel_initializer = initializer)

		x1 = tf.reshape(x1, (-1, 4, 4, 1024))

		x1 = tf.layers.batch_normalization(x1, training = training)

		x1 = tf.maximum(alpha*x1, x1) # Shape = 4x4x1024 

		# Dropout line here



		# Layer 2

		x2 = tf.layers.conv2d_transpose(inputs = x1, filters = 512, kernel_size = 5, strides = 2, padding = "same", kernel_initializer = initializer)

		x2 = tf.layers.batch_normalization(x2, training = training)

		x2 = tf.maximum(alpha*x2, x2) # 8x8x512 

		# Dropout line here



		# Layer 3

		x3 = tf.layers.conv2d_transpose(inputs = x2, filters = 256, kernel_size = 5, strides = 2, padding = "same", kernel_initializer = initializer)

		x3 = tf.layers.batch_normalization(x3, training = training)

		x3 = tf.maximum(alpha*x3, x3) # 16x16x256

		# Dropout line here



		# Layer 4

		x4 = tf.layers.conv2d_transpose(inputs = x3, filters = 128, kernel_size = 5, strides = 2, padding = "same", kernel_initializer = initializer)

		x4 = tf.layers.batch_normalization(x4, training = training)

		x4 = tf.maximum(alpha*x4, x4) # 32x32x128

		# Dropout line here



		# Output layer

		logits = tf.layers.conv2d_transpose(inputs = x4, filters = output_dim, kernel_size = 5, strides = 2, padding = "same", kernel_initializer = initializer)

		out = tf.tanh(logits) # 64x64x3



		return out
# Discriminator network

def discriminator(x, reuse = False, alpha = 0.2, initializer = tf.random_normal_initializer(0, 0.02)):

	with tf.variable_scope("discriminator", reuse = reuse):

		# Input layer is 64x64x3

		x1 = tf.layers.conv2d(inputs = x, filters = 64, kernel_size = 5, strides = 2, padding = "same", kernel_initializer = initializer)

		relu1 = tf.maximum(alpha*x1, x1) # 32x32x64

		# Dropout line here



		# Layer 2

		x2 = tf.layers.conv2d(inputs = relu1, filters = 128, kernel_size = 5, strides = 2, padding = "same", kernel_initializer = initializer)

		bn2 = tf.layers.batch_normalization(inputs = x2, training = True)

		relu2 = tf.maximum(alpha*bn2, bn2) # 16x16x128

		# Dropout line here



		# Layer 3

		x3 = tf.layers.conv2d(inputs = relu2, filters = 256, kernel_size = 5, strides = 2, padding = "same", kernel_initializer = initializer)

		bn3 = tf.layers.batch_normalization(inputs = x3, training = True)

		relu3 = tf.maximum(alpha*bn3, bn3) # 8x8x256

		# Dropout line here



		# Flatten it

		flat = tf.reshape(relu3, (-1, 8*8*256))

		logits = tf.layers.dense(inputs = flat, units = 1, kernel_initializer = initializer)

		out = tf.sigmoid(logits)



		return out, logits
# Model loss function

def model_loss(input_real, input_z, output_dim, alpha = 0.2, smooth = 0.1, initializer = tf.random_normal_initializer(0, 0.02), drop_threshold = 0.2):

	"""

	Get the loss for discriminator and generator

	:param input_real: images from the real dataset

	:param input_z: Z input

	:param out_channel_dim: the number of channels in the output image

	:param alpha: alpha in Leaky ReLU

	:param smooth: range (0.0, 1.0) - used for label smoothing

	:param initializer: kernel initializer for both generator and discriminator network

	:param drop_threshold: threshold probability for uniform distribution that converts some of the labels of real images from 1 to 0

	:return: A tuple of (discriminator loss, generator loss)

	"""



	g_model = generator(input_z, output_dim, alpha = alpha, initializer = initializer) # Fake input to discriminator

	d_model_real, d_logits_real = discriminator(input_real, alpha = alpha, initializer = initializer) # Training on real images

	d_model_fake, d_logits_fake = discriminator(g_model, alpha = alpha, reuse = True, initializer = initializer) # Training on fake images



	#d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real, labels = tf.ones_like(d_model_real)*(1 - smooth)))

	random_drop = tf.cast(tf.random.uniform(shape = tf.shape(d_model_real), minval = 0.0, maxval = 1.0) >= drop_threshold, tf.float32)

	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real, labels = tf.ones_like(d_model_real)*random_drop*(1 - smooth)))

	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, labels = tf.zeros_like(d_model_fake)))

	g_loss 		= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, labels = tf.ones_like(d_model_fake)))



	d_loss = d_loss_real + d_loss_fake



	return d_loss, g_loss
# Optimizers

def model_opt(d_loss, g_loss, learning_rate, beta1):

	"""

	Get optimization operations

	:param d_loss: Discriminator loss Tensor

	:param g_loss: Generator loss Tensor

	:param learning_rate: Learning Rate Placeholder

	:param betq: The exponential decay rate for the 1st moment in the optimizer

	:return: A tuple of (discriminator training operation, generator training operation)

	"""



	# Get weights and bias to update

	t_vars = tf.trainable_variables()

	d_vars = [var for var in t_vars if var.name.startswith("discriminator")]

	g_vars = [var for var in t_vars if var.name.startswith("generator")]



	# Optimize

	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

		d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(d_loss, var_list = d_vars)

		g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(g_loss, var_list = g_vars)



	return d_train_opt, g_train_opt
# Building the model

class GAN:

	def __init__(self, real_size, z_size, learning_rate, alpha = 0.2, beta1 = 0.5, smooth = 0.1, initializer = tf.random_normal_initializer(0, 0.02), drop_threshold = 0.2):

		"""

		:param real_size: real image dimension (H, W, channels)

		:param z_size: Latent space dimension (integer)

		:param learning_rate: Learning rate for optimizer

		:param alpha: Leaky ReLU parameter

		:param beta1: The exponential decay rate for the 1st moment in the optimizer

		:param smooth: Label smoothing, range (0.0, 1.0)

		:param initializer: kernel initializer for the generator and discrminator network

		"""



		tf.reset_default_graph()

		self.input_real, self.input_z = model_inputs(real_size, z_size)

		self.d_loss, self.g_loss = model_loss(self.input_real, self.input_z, real_size[2], alpha = alpha, smooth = smooth, initializer = initializer, drop_threshold = drop_threshold)

		self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1)
# Function to display generated images

def view_samples(epoch, samples, nrows, ncols, figsize = (10, 10)):

	fig, axes = plt.subplots(figsize = figsize, nrows = nrows, ncols = ncols, sharey = True, sharex = True)



	for ax, img in zip(axes.flatten(), samples[epoch]):

		ax.axis("off")

		img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)

		ax.set_adjustable("box-forced")

		im = ax.imshow(img, aspect = "equal")



	plt.subplots_adjust(wspace = 0, hspace = 0)

	return fig, axes
# Data preprocessing

# Rescale to [-1, 1] as output of our generator is in that range

def scale(x, feature_range = (-1, 1)):

	# scale to (0, 1)

	x = ((x - x.min())/(255 - x.min()))



	# scale to feature_range

	min, max = feature_range

	x = x*(max - min) + min

	return x
class Dataset:

	# initialize the attributes of the class

	def __init__(self, images, names, val_frac = 0.5, shuffle = False, scale_func = None):

		# Split the filenames into train and valid set

		split_idx = int(len(names)*(1 - val_frac))

		self.train_x, self.valid_x = images[:split_idx, :, :, :], images[split_idx:, :, :, :] # Split the images

		self.train_y, self.valid_y = names[:split_idx], names[split_idx:] # Split the names accordingly

		self.shuffle = shuffle



		# Scaling function

		if scale_func is None:

			self.scaler = scale

		else:

			self.scaler = scale_func





	def batches(self, batch_size):

		# Shuffle if True

		if self.shuffle:

			index = np.arange(len(self.train_x))

			np.random.shuffle(index)

			self.train_x = self.train_x[index]

			self.train_y = self.train_y[index]



		# Provide the minibatch

		n_batches = len(self.train_y)//batch_size

		for ii in range(0, len(self.train_y), batch_size):

			x = self.train_x[ii: ii + batch_size]

			y = self.train_y[ii: ii + batch_size]



			yield self.scaler(x), y
# Function to train the network

def train(net, dataset, batch_size, epochs, figsize = (12, 12), print_every = 10, show_every = 100):

#	saver = tf.train.Saver()

#	sample_z = np.random.uniform(-1, 1, size = (72, z_size))

	sample_z = np.random.normal(size = (72, z_size))



	samples, losses = [], []

	steps = 0



	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		for e in range(epochs):

			for x, y in dataset.batches(batch_size):

				steps += 1



				# Sample random noise for G

#				batch_z = np.random.uniform(-1, 1, size = (batch_size, z_size))

				batch_z = np.random.normal(size = (batch_size, z_size))



				# Run optimizers

				_ = sess.run(net.d_opt, feed_dict = {net.input_real: x, net.input_z: batch_z})

				_ = sess.run(net.g_opt, feed_dict = {net.input_z: batch_z, net.input_real: x})





				# Calculate the generative and discriminator model loss after every n1 steps

				if steps % print_every == 0:

					# At the end of each epoch, get the losses and print them out

					train_loss_d = net.d_loss.eval({net.input_z: batch_z, net.input_real: x})

					train_loss_g = net.g_loss.eval({net.input_z: batch_z})



					print("Epoch {}/{}...".format(e+1, epochs),

						"Step {}".format(steps),

						"Discriminator Loss: {:.4f}...".format(train_loss_d),

						"Generator Loss: {:.4f}".format(train_loss_g))



					# Save losses to view after training

					losses.append((train_loss_d, train_loss_g))



				# Show the generated picture after every n2 steps

				if steps % show_every == 0:

					gen_samples = sess.run(generator(net.input_z, 3, reuse = True, training = False), feed_dict = {net.input_z: sample_z})

					samples.append(gen_samples)

					_ = view_samples(-1, samples, 1, 6, figsize = figsize)

					plt.show()



#		saver.save(sess, "./checkpoints/dcgan_generator.ckpt")



#	with open("samples.pkl", "wb") as f:

#		pkl.dump(samples, f)



	return losses, samples
# Hyper parameters

real_size = (64, 64, 3)

z_size = 100

learning_rate = 0.0002

batch_size = 128

epochs = 70

alpha = 0.2

beta1 = 0.5

smooth = 0.1

drop_threshold = 0.1

initializer = tf.random_normal_initializer(0, 0.02)



# Create the network

net = GAN(real_size, z_size, learning_rate, alpha = alpha, beta1 = beta1, smooth = smooth, initializer = initializer, drop_threshold = drop_threshold)
# Training

dataset = Dataset(images, names, val_frac = 0.3, shuffle = True)

losses, samples = train(net, dataset, batch_size, epochs)
# Visualization of training losses

fig, ax = plt.subplots()

losses = np.array(losses)

plt.plot(losses.T[0], label = "Discriminator", alpha = 0.5)

plt.plot(losses.T[1], label = "Generator", alpha = 0.5)

plt.ylim([0, 4])

plt.title("Training losses")

plt.legend()
_ = view_samples(-1, samples, 4, 4, figsize = (7, 7))