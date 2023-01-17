from keras.applications import VGG16

from keras import backend as K

import numpy as np

import matplotlib.pyplot as plt

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input, decode_predictions



model = VGG16(weights="imagenet")
img_path = "../input/tiger.jpg"

img1 = image.load_img(img_path)

plt.imshow(img1)



img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)
preds = model.predict(x)

print("Prediction:", decode_predictions(preds, top=3)[0])
np.argmax(preds[0])
# Feature map for the last convolutional layer in VGG16

tiger_output = model.output[:, 292]

last_conv_layer = model.get_layer("block5_conv3")



grads = K.gradients(tiger_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))



iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])



pooled_grads_value, conv_layer_output_value = iterate([x])



for i in range(512):

    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)

heatmap /= np.max(heatmap)

plt.matshow(heatmap)
import cv2 as cv



img = cv.imread(img_path)

heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)

heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img



save_img_path = "heatmap_tiger.jpg"

cv.imwrite(save_img_path, superimposed_img)

img1 = image.load_img(save_img_path)

plt.imshow(img1)