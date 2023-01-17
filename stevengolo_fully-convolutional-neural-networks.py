# Load packages

import urllib

import warnings



import numpy as np 

import matplotlib.pyplot as plt

import tensorflow as tf



from scipy.io import loadmat



from skimage.io import imread

from skimage.transform import resize



from tensorflow.keras import layers

from tensorflow.keras.applications.imagenet_utils import preprocess_input

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import Convolution2D

from tensorflow.keras.models import Model



# Set seed for the session

np.random.seed(42)
# Load a pre-trained ResNet50 model

# We use include_top=False for now

base_model = ResNet50(include_top=False)



print(f'Shape of the output of the model: {base_model.output_shape}.')
print(base_model.summary())
# Get the last layer of the ResNet model

res5c = base_model.layers[-1]
print(f'Type of res5c: {type(res5c)}, Output shape of res5c: {res5c.output_shape}.')
# Define the custom Softmax layer

class SoftmaxMap(layers.Layer):

    def __init__(self, axis=-1, **kwargs):

        self.axis = axis

        super(SoftmaxMap, self).__init__(**kwargs)

        

    def build(self, input_shape):

        pass

    

    def call(self, x, mask=None):

        """This function is very similar to the regular Softmax but

        we accept x.shape == (batch_size, w, h, n_classes)

        which is not the case in Keras by default.

        Note also that we substract the logits by their maximum to

        make the softmax numerically stable.

        """

        e = tf.exp(x - tf.math.reduce_max(x, axis=self.axis, keepdims=True))

        s = tf.math.reduce_sum(e, axis=self.axis, keepdims=True)

        return e / s

    

    def get_output_shape_for(self, input_shape):

        return input_shape
n_samples, w, h, n_classes = 10, 3, 4, 5

random_data = np.random.randn(n_samples, w, h, n_classes).astype('float32')



print(f'Shape of random_data: {random_data.shape}.')
random_data[0].sum(axis=-1)
softmaxMap = SoftmaxMap()

softmax_mapped_data = softmaxMap(random_data).numpy()



print(f'Shape of softmax_mapped_data: {softmax_mapped_data.shape}.')
softmax_mapped_data[0].sum(axis=-1)
np.alltrue(random_data[0].argmax(axis=-1) == softmax_mapped_data[0].argmax(axis=-1))
# Define a Fully Convolutional ResNet

input_tensor = base_model.layers[0].input



# Take the output of the last layer of the ConvNet model

output_tensor = base_model.layers[-1].output



# A 1x1 convolution, with 1000 output channels, one per class

output_tensor = Convolution2D(1000, (1, 1), name='Conv1000')(output_tensor)



# Softmax on last axis of tensor to normalize the class predictions in each spatial area

output_tensor = SoftmaxMap(axis=-1)(output_tensor)



# Define model

fully_conv_resnet = Model(inputs=input_tensor, outputs=output_tensor)
prediction_maps = fully_conv_resnet(np.random.randn(1, 200, 300, 3)).numpy()

print(f'Shape of the predictions: {prediction_maps.shape}.')
prediction_maps.sum(axis=-1)
# Load weights and biases

complete_model = ResNet50(include_top=True)

W = complete_model.layers[-1].get_weights()[0]

B = complete_model.layers[-1].get_weights()[1]



last_layer = fully_conv_resnet.layers[-2]



print(f'Shape of the weights of the last layer from the ResNet50 model: {W.shape}.')

print(f'Shape of the weights of the last convolutional layer: {last_layer.get_weights()[0].shape}.')
# Reshape the weights

W_reshaped = W.reshape((1, 1, 2048, 1000))



# Set the convolution layer weights

last_layer.set_weights([W_reshaped, B])
def forward_pass_resize(img_path, img_size):

    img_raw = imread(img_path)

    img = resize(img_raw, img_size, mode='reflect', preserve_range=True)

    img = preprocess_input(img[np.newaxis])

    print(f'Shape of the raw image: {img_raw.shape}.')

    print(f'Shape of the reshaped image: {img.shape}.')

    

    prediction_map = fully_conv_resnet(img).numpy()

    return prediction_map
IMG_URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Female_Black_Labrador_Retriever.jpg/1280px-Female_Black_Labrador_Retriever.jpg'

urllib.request.urlretrieve(IMG_URL, 'dog.jpg')

output = forward_pass_resize('dog.jpg', (800, 600))

print(f'Shape of the prediction map: {output.shape}.')
# Get synset data

SYNSET_DAT = 'https://github.com/m2dsupsdlclass/lectures-labs/raw/master/labs/05_conv_nets_2/data/meta_clsloc.mat'

urllib.request.urlretrieve(SYNSET_DAT, 'synset_dat.mat')
# Load synsets

synsets = loadmat('synset_dat.mat')['synsets'][0]

synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]], key=lambda v:v[1])



corr = {}

for j in range(1000):

    corr[synsets_imagenet_sorted[j][0]] = j



corr_inv = {}

for j in range(1, 1001):

    corr_inv[corr[j]] = j



def depthfirstsearch(id_, out=None):

    if out is None:

        out = []

    if isinstance(id_, int):

        pass

    else:

        id_ = next(int(s[0]) for s in synsets if s[1][0] == id_)

        

    out.append(id_)

    children = synsets[id_ - 1][5][0]

    for c in children:

        depthfirstsearch(int(c), out)

    return out



def synset_to_dfs_ids(synset):

    ids = [x for x in depthfirstsearch(synset) if x <= 1000]

    ids = [corr[x] for x in ids]

    return ids



def id_to_words(id_):

    return synsets[corr_inv[id_] - 1][2][0]
synset_dog = 'n02084071'

idx = synset_to_dfs_ids(synset_dog)

print(f'Number of dog classes ids: {len(idx)}.')
def build_heatmap(img_path, synset, size):

    """Build a heatmap 

    :param img_path: path of the input image, str

    :param synset: synset to find in the image, str

    :param size: size of the reshaped image, tuple

    """

    prediction_map = forward_pass_resize(img_path, size)

    

    class_ids = synset_to_dfs_ids(synset)

    class_ids = np.array([id_ for id_ in class_ids if id_ is not None])

    

    each_dog_proba_map = prediction_map[0, :, :, class_ids]

    # This style of indexing a tensor by an other array has the following shape effect:

    # (H, W, 1000) indexed by (118) => (118, H, W)

    any_dog_proba_map = each_dog_proba_map.sum(axis=0)

    return any_dog_proba_map
def display_img_heatmap(img_path, heatmap):

    """Display the image and the heatmap side by side

    """

    img = imread(img_path)

    

    plt.figure(figsize=(12, 8))

    

    plt.subplot(1, 2, 1)

    plt.imshow(img)

    plt.title(f'Image {img.shape}')

    plt.axis('off')

    

    plt.subplot(1, 2, 2)

    plt.imshow(heatmap, interpolation='nearest', cmap='viridis')

    plt.title(f'Heatmap {heatmap.shape}')

    plt.axis('off')
# (200, 320)

heatmap_200x320 = build_heatmap('dog.jpg', synset_dog, (200, 320))

display_img_heatmap('dog.jpg', heatmap_200x320)
# (400, 640)

heatmap_400x640 = build_heatmap('dog.jpg', synset_dog, (400, 640))

display_img_heatmap('dog.jpg', heatmap_400x640)
# (800, 1280)

heatmap_800x1280 = build_heatmap('dog.jpg', synset_dog, (800, 1280))

display_img_heatmap('dog.jpg', heatmap_800x1280)
# (1600, 2560)

heatmap_1600x2560 = build_heatmap('dog.jpg', synset_dog, (1600, 2560))

display_img_heatmap('dog.jpg', heatmap_1600x2560)
# We resize each of the heatmap to the larger one.

heatmap_200x320_r = resize(heatmap_200x320, (50, 80), mode='reflect',

                           preserve_range=True, anti_aliasing=True)

heatmap_400x640_r = resize(heatmap_400x640, (50, 80), mode='reflect',

                           preserve_range=True, anti_aliasing=True)

heatmap_800x1280_r = resize(heatmap_800x1280, (50, 80), mode='reflect',

                            preserve_range=True, anti_aliasing=True)
# Arithmetic average

heatmap = (heatmap_200x320_r + heatmap_400x640_r + heatmap_800x1280_r + heatmap_1600x2560) / 4

display_img_heatmap('dog.jpg', heatmap)
# Geometric average

heatmap = np.power(heatmap_200x320_r * heatmap_400x640_r * heatmap_800x1280_r * heatmap_1600x2560, 0.25)

display_img_heatmap('dog.jpg', heatmap)