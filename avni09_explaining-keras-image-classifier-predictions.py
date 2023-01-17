from PIL import Image

from IPython.display import display

import numpy as np



# you may want to keep logging enabled when doing your own work

import logging

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR) # disable Tensorflow warnings for this tutorial

import warnings

warnings.simplefilter("ignore") # disable Keras warnings for this tutorial

import keras

from keras.applications import mobilenet_v2



import eli5
model = mobilenet_v2.MobileNetV2(include_top=True, weights='imagenet', classes=1000)



# check the input format

print(model.input_shape)

dims = model.input_shape[1:3] # -> (height, width)

print(dims)
image_uri = '/Users/avnig/XAI/eli5/notebooks/imagenet-samples/cat_dog.jpg'



# check the image with Pillow

im = Image.open(image_uri)



display(im)
# we could resize the image manually

# but instead let's use a utility function from `keras.preprocessing`

# we pass the required dimensions as a (height, width) tuple

im = keras.preprocessing.image.load_img(image_uri, target_size=dims) # -> PIL image

print(im)

display(im)
# we use a routine from `keras.preprocessing` for that as well

# we get a 'doc', an object almost ready to be inputted into the model



doc = keras.preprocessing.image.img_to_array(im) # -> numpy array

print(type(doc), doc.shape)
# dimensions are looking good

# except that we are missing one thing - the batch size



# we can use a numpy routine to create an axis in the first position

doc = np.expand_dims(doc, axis=0)

print(type(doc), doc.shape)
# `keras.applications` models come with their own input preprocessing function

# for best results, apply that as well



# mobilenetv2-specific preprocessing

# (this operation is in-place)

mobilenet_v2.preprocess_input(doc)

print(type(doc), doc.shape)
# take back the first image from our 'batch'

image = keras.preprocessing.image.array_to_img(doc[0])

print(image)

display(image)
# make a prediction about our sample image

predictions = model.predict(doc)

print(type(predictions), predictions.shape)
# check the top 5 indices

# `keras.applications` contains a function for that



top = mobilenet_v2.decode_predictions(predictions)

top_indices = np.argsort(predictions)[0, ::-1][:5]



print(top)

print(top_indices)
# we need to pass the network

# the input as a numpy array

eli5.show_prediction(model, doc)
eli5.show_prediction(model, doc, image=image)
cat_idx = 282 # ImageNet ID for "tiger_cat" class, because we have a cat in the picture

eli5.show_prediction(model, doc, targets=[cat_idx]) # pass the class id
cat_idx = 285 # 'Egyptian cat'

display(eli5.show_prediction(model, doc, targets=[cat_idx]))
# we could use model.summary() here, but the model has over 100 layers. 

# we will only look at the first few and last few layers



head = model.layers[1:]

tail = model.layers[:]



def pretty_print_layers(layers):

    for l in layers:

        info = [l.name, type(l).__name__, l.output_shape, l.count_params()]

        pretty_print(info)



def pretty_print(lst):

    s = ',\t'.join(map(str, lst))

    print(s)



pretty_print(['name', 'type', 'output shape', 'param. no'])

print('-'*100)

pretty_print([model.input.name, type(model.input), model.input_shape, 0])

pretty_print_layers(head)

print()

print('...')

print()

pretty_print_layers(tail)
for l in ['block_2_expand', 'block_9_expand', 'block_16_expand']:

    print(l)

    display(eli5.show_prediction(model, doc, layer=l)) # we pass the layer as an argument
expl = eli5.explain_prediction(model, doc)
print(expl)
# we can access the various attributes of a target being explained

print((expl.targets[0].target, expl.targets[0].score, expl.targets[0].proba))
image = expl.image

heatmap = expl.targets[0].heatmap



display(image) # the .image attribute is a PIL image

print(heatmap) # the .heatmap attribute is a numpy array
heatmap_im = eli5.formatters.image.heatmap_to_image(heatmap)

heatmap_im = eli5.formatters.image.expand_heatmap(heatmap, image, resampling_filter=Image.BOX)

display(heatmap_im)
I = eli5.format_as_image(expl)

display(I)
import matplotlib.cm



I = eli5.format_as_image(expl, alpha_limit=1.0, colormap=matplotlib.cm.cividis)

display(I)
# first check the explanation *with* softmax

print('with softmax')

display(eli5.show_prediction(model, doc))





# remove softmax

l = model.get_layer(index=-1) # get the last (output) layer

l.activation = keras.activations.linear # swap activation



# save and load back the model as a trick to reload the graph

model.save('tmp_model_save_rmsoftmax') # note that this creates a file of the model

model = keras.models.load_model('tmp_model_save_rmsoftmax')



print('without softmax')

display(eli5.show_prediction(model, doc))
from keras.applications import nasnet



model2 = nasnet.NASNetMobile(include_top=True, weights='imagenet', classes=1000)



# we reload the image array to apply nasnet-specific preprocessing

doc2 = keras.preprocessing.image.img_to_array(im)

doc2 = np.expand_dims(doc2, axis=0)

nasnet.preprocess_input(doc2)



print(model.name)

# note that this model is without softmax

display(eli5.show_prediction(model, doc))

print(model2.name)

display(eli5.show_prediction(model2, doc2))