import os
from os.path import join


hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/hot_dog'

hot_dog_paths = [join(hot_dog_image_dir,filename) for filename in 
                            ['1000288.jpg',
                             '127117.jpg']]

not_hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/not_hot_dog'
not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                            ['823536.jpg',
                             '99890.jpg']]

img_paths = hot_dog_paths + not_hot_dog_paths
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from learntools.deep_learning.decode_predictions import decode_predictions

image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)


my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)

most_likely_labels = decode_predictions(preds, top=3)
from IPython.display import Image, display

for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])
# first we will create a function which will return true or false depending on whether the image 
# hot dog or not.

def is_hot_dog(preds):      
    decoded = decode_predictions(preds, top=1)
    labels =[]
    for arr in decoded:
        label = arr[0][1]
        labels.append(label)

    is_hot_dog = []
    for val in labels:
        if val=='hotdog':
            is_hot_dog.append(True)
        else:
            is_hot_dog.append(False)
    return is_hot_dog
    pass

# Now this function will calculate accuracy
def calc_accuracy(model, paths_to_hotdog_images, paths_to_other_images):
    actual_labels = []
    for i in range(len(paths_to_hotdog_images)):
        actual_labels.append(True)
    for i in range(len(paths_to_other_images)):
        actual_labels.append(False)
        pass
    
    paths = paths_to_hotdog_images+paths_to_other_images
    test_data = read_and_prep_images(paths)
    preds = my_model.predict(test_data)
    predicted_labels = is_hot_dog(preds)
    
    corrects = 0
    for i, label in enumerate(predicted_labels):
        if label == actual_labels[i]:
            corrects+=1
    accuracy = corrects/len(predicted_labels)
    return accuracy
    pass

# Code to call calc_accuracy.  my_model, hot_dog_paths and not_hot_dog_paths were created in the setup code
my_model_accuracy = calc_accuracy(my_model, hot_dog_paths, not_hot_dog_paths)
print("Fraction correct in our small test set: {}".format(my_model_accuracy))


