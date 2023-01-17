from os.path import join

# dogs images data is in this directory
image_dir = '../input/dog-breed-identification/'

#Here is a few image files to test. We put the file paths in a list. 
image_filenames = ['0c8fe33bd89646b678f6b2891df8a1c6.jpg',
                   '0c3b282ecbed1ca9eb17de4cb1b6e326.jpg',
                   '04fb4d719e9fe2b6ffe32d9ae7be8a22.jpg',
                   '0e79be614f12deb4f7cae18614b7391b.jpg']

#Then we use the join function from python's "os.path" to append the file name to the directory.
#The end result is a list of paths to image files
img_paths = [join(image_dir, filename) for filename in image_filenames ]
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#The model we'll use was trained with 224x224 resolution images so we'll make them
#have the same resolution here.
image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    #load the images using the load_img() function.
    #We have a few images so we keep them in a list for now using a list comprehension. 
    #The target size argument specifies the size or pixel resolution we want the images to be and when we model with them.
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]

    #convert each image into an array using the img_to_array() function.
    #the img_to_array() function creates 3d tensor for each image combining multiple images 
    #cause us to stack those in a new dimension so we end up with a 4 dimensional tensor or array
    img_array = np.array([img_to_array(img) for img in imgs])

    #preprocess_input() function does some arithmetic on the pixel values.
    #The outut values became between minus 1 and 1. This was done when a model was first built so we have to do it again here to be consistent.
    #It returns preprocessed numpy.array or a tf.Tensor with type float32. 
    #The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.
    output = preprocess_input(img_array)
    
    return(output)
from tensorflow.keras.applications import ResNet50

#We'll use a type of model called the ResNet 50 model.
#We give it an argument specify the file path where we have stored the values in the convolutional filters
#Return value: a Keras model instance.
my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

#call that function we just wrote before to read and preprocess our data
test_data = read_and_prep_images(img_paths)

#get predictions by calling the predict() method of our model
#Returns array of likeness rates with known categories (i.e. dogs, cats, coffe, etc.) for each input image
preds = my_model.predict(test_data)

preds
from learntools.deep_learning.decode_predictions import decode_predictions

#call the function with the prediction results and tell it to give us the top three probabilities for each photo
most_likely_labels = decode_predictions(preds, top=3, class_list_path='../input/resnet50/imagenet_class_index.json')
from IPython.display import Image, display

for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])