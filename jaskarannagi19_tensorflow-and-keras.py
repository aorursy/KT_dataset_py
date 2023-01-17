import os
from PIL import Image
from os.path import join
# Kumar -> Changing image_dir and img_paths -> Unable to use the new files and reverted
#image_dir = '../input/dog-breed-identification/train/'
image_dir = '../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test'
img_paths = [join(image_dir, filename) for filename in 
                           #['0246f44bb123ce3f91c939861eb97fb7.jpg',
                          #'84728e78632c0910a69d33f82e62638c.jpg',
                           # '8825e914555803f4c67b26593c9d5aff.jpg',
                            #'91a5e8db15bccfb6cfa2df5e8b95ec03.jpg']]
             
                           ['A_test.jpg',
                            'C_test.jpg',
                           'E_test.jpg',
                           'G_test.jpg']]
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)
from tensorflow.python.keras.applications import ResNet50

my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)
# Kumar - created a png file but am unable to locate the model.png file
from tensorflow.python.keras.utils import plot_model
plot_model(my_model, to_file='../working/model.png')
print(os.listdir('../working'))



plot_model(my_model, to_file='model.png')


from IPython.display import Image
Image("../working/model.png")
# Kumar - Visualizing the model from https://keras.io/visualization/ - This worked
#from IPython.display import SVG
#from tensorflow.python.keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(my_model).create(prog='dot', format='svg'))
# Kumar - Trying suggestion from Kaggle learn forums
my_model.summary()
# Kumar - trying ANN visualization as per https://towardsdatascience.com/visualizing-artificial-neural-networks-anns-with-just-one-line-of-code-b4233607209e
# Not working - How to install ann_viz? 
#ann_viz(my_model, view=True, filename='network.gv', title='MyNeural Network')
#pip install ann_visualizer
import sys
# Add directory holding utility functions to path to allow importing
sys.path.append('/kaggle/input/python-utility-code-for-deep-learning-exercises/utils')
from decode_predictions import decode_predictions

from IPython.display import Image, display

most_likely_labels = decode_predictions(preds, top=3, class_list_path='../input/resnet50/imagenet_class_index.json')

for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])