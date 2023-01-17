from keras.preprocessing import image

from keras.applications import inception_v3

from keras import backend as K

from keras.applications.inception_v3 import decode_predictions

from PIL import Image

from matplotlib import pyplot as plt

import numpy as np

import cv2
# Load pre-trained image recognition model

model = inception_v3.InceptionV3()



# Grab a reference to the first and last layer of the neural net

model_input_layer = model.layers[0].input

model_output_layer = model.layers[-1].output



# Choose an ImageNet object to fake

# The list of classes is available here: https://gist.github.com/ageitgey/4e1342c10a71981d0b491e1b8227328b

# Class #859 is "toaster"

object_type_to_fake = 859
# Load the image to hack

img = image.load_img("../input/cat.png", target_size=(299, 299))

original_image = image.img_to_array(img)

img
# Scale the image so all pixel intensities are between [-1, 1] as the model expects

original_image /= 255.

original_image -= 0.5

original_image *= 2.



# Add a 4th dimension for batch size (as Keras expects)

original_image = np.expand_dims(original_image, axis=0)
_, category, confidence = decode_predictions(model.predict(original_image), top=1)[0][0]

print ("Original Image is a %s with %f confidence."%(category, confidence))
# Pre-calculate the maximum change we will allow to the image

# We'll make sure our hacked image never goes past this so it doesn't look funny.

# A larger number produces an image faster but risks more distortion.

max_change_above = original_image + 0.01

max_change_below = original_image - 0.01



# Create a copy of the input image to hack on

hacked_image = np.copy(original_image)



# How much to update the hacked image in each iteration

learning_rate = 10



# Define the cost function.

# Our 'cost' will be the likelihood out image is the target class according to the pre-trained model

cost_function = model_output_layer[0, object_type_to_fake]



# We'll ask Keras to calculate the gradient based on the input image and the currently predicted class

# In this case, referring to "model_input_layer" will give us back image we are hacking.

gradient_function = K.gradients(cost_function, model_input_layer)[0]



# Create a Keras function that we can call to calculate the current cost and gradient

grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])
cost = 0.0

n = 0

# In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more

# until it gets to at least 99% confidence

while 1:

    # Check how close the image is to our target class and grab the gradients we

    # can use to push it one more step in that direction.

    # Note: It's really important to pass in '0' for the Keras learning mode here!

    # Keras layers behave differently in prediction vs. train modes!

    cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

    if cost > 0.99:

        break



    # Move the hacked image one step further towards fooling the model

    hacked_image += gradients * learning_rate



    # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image

    hacked_image = np.clip(hacked_image, max_change_below, max_change_above)

    hacked_image = np.clip(hacked_image, -1.0, 1.0)

    if n % 50 == 0:

        print("Model's predicted likelihood that the image is a toaster: {:.8}%".format(cost * 100))

    n += 1
# De-scale the image's pixels from [-1, 1] back to the [0, 255] range

hacked_img = hacked_image.copy()[0]

hacked_img /= 2.

hacked_img += 0.5

hacked_img *= 255.



original_image /= 2.

original_image += 0.5

original_image *= 255.
# New image is misclassified as a toster.

_, category, confidence = decode_predictions(model.predict(hacked_image), top=1)[0][0]

print ("Hacked Image is a %s with %f confidence."%(category, confidence))
# show the difference between original image and hacked image.

plt.figure(figsize=(20,20))

plt.imshow(np.hstack([img, img-hacked_img, hacked_img]).astype(np.uint8), interpolation='nearest')

plt.show()
# Defense by using GaussianBlur.

kernel_size = (5, 5)

sigma = 1.0

image_GaussianBlur = cv2.GaussianBlur(hacked_img, kernel_size, sigma)

img_GaussianBlur = image_GaussianBlur.copy()

img_GaussianBlur /= 255

img_GaussianBlur -= 0.5

img_GaussianBlur *= 2
_, category, confidence = decode_predictions(model.predict(np.expand_dims(img_GaussianBlur, axis=0)), top=1)[0][0]

print ("After GaussianBlur, Image is a %s with %f confidence."%(category, confidence))
# Let's check the image. 

plt.figure(figsize=(10,10))

plt.imshow(np.hstack([hacked_img, image_GaussianBlur]).astype(np.uint8), interpolation='nearest')

plt.show()