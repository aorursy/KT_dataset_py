import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from os.path import join

image_dir = '../input/images/'
img_paths = [join(image_dir, filename) for filename in 
                           ['img1.jpg',
                            'img2.jpg']]

image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)
from tensorflow.python.keras.applications import ResNet50

my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
print (test_data.shape)
preds = my_model.predict(test_data)
import sys
# Add directory holding utility functions to path to allow importing
sys.path.append('/kaggle/input/python-utility-code-for-deep-learning-exercises/utils')
from decode_predictions import decode_predictions

from IPython.display import Image, display

most_likely_labels = decode_predictions(preds, top=3, class_list_path='../input/resnet50/imagenet_class_index.json')

for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])