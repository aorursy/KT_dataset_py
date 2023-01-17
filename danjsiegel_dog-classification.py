img_paths = []

img_paths.append('../input/mister-banks-picture/mr_banks.jpg')
import numpy as np

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



image_size = 224



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    img_array = np.array([img_to_array(img) for img in imgs])

    output = preprocess_input(img_array)

    return(output)
from tensorflow.python.keras.applications import ResNet50



my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

test_data = read_and_prep_images(img_paths)

preds = my_model.predict(test_data)
from learntools.deep_learning.decode_predictions import decode_predictions

from IPython.display import Image, display



most_likely_labels = decode_predictions(preds, top=3, class_list_path='../input/resnet50/imagenet_class_index.json')



for i, img_path in enumerate(img_paths):

    display(Image(img_path))

    print(most_likely_labels[i])