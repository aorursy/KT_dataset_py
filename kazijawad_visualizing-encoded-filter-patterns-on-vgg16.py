from keras.applications import VGG16

from keras import backend as K

import numpy as np

import matplotlib.pyplot as plt



model = VGG16(weights="imagenet", include_top=False)



layer_name = "block3_conv1"

filter_index = 0



layer_output = model.get_layer(layer_name).output

loss = K.mean(layer_output[:, :, :, filter_index])



grads = K.gradients(loss, model.input)[0]

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)



iterate = K.function([model.input], [loss, grads])

loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])



# Generate blank gray image with noise

input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

step = 1.



for i in range(40):

    # Calculate loss and gradient values

    loss_value, grads_value = iterate([input_img_data])

    

    # Adjust the image for maximizing loss

    input_img_data += grads_value * step
# Converts a tensor into an image

def deprocess_image(x):

    x -= x.mean()

    x /= (x.std() + 1e-5)

    x *= 0.1

    x += 0.5

    x = np.clip(x, 0, 1)

    x *= 255

    x = np.clip(x, 0, 255).astype("uint8")

    return x
def generate_pattern(layer_name, filter_index, size=150):

    layer_output = model.get_layer(layer_name).output

    loss = K.mean(layer_output[:, :, :, filter_index])

    

    grads = K.gradients(loss, model.input)[0]

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])

    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    step = 1

    

    for i in range(40):

        loss_value, grads_value = iterate([input_img_data])

        input_img_data += grads_value * step

        img = input_img_data[0]

        

    return deprocess_image(img)
# Filter pattern for layer block3_conv1 at the zeroth channel

plt.imshow(generate_pattern("block3_conv1", 1))