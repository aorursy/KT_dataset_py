from os.path import join



image_dir = '../input/user-dogs/' #filepath ../input/dog-breed-identification/train/

img_paths = [join(image_dir, filename) for filename in 

                           ['stormTEST.jpg']] #joined with image director to make path to image file that is being predicted
import numpy as np

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



image_size = 224 #size of image that model is being trained on



def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size): #normalize data

    #target_size specifies the size of the image that we want when we model with them

    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths] #loads images using load_img since it can classify multiple images, using list comprehension

    img_array = np.array([img_to_array(img) for img in imgs]) #convert each image into an array. Storing images in 3D tensor, which are stacked in an array, making them 4D

    output = preprocess_input(img_array) #does epic math on pixels - making sure pixel values are between -1 and 1 - done for consistency

    return(output)
from tensorflow.python.keras.applications import ResNet50



my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5') #specifying the model in this case, ResNet50. The file path points to where the values of the convolutional filters are

test_data = read_and_prep_images(img_paths) #read and pre-process model

preds = my_model.predict(test_data) #get predictions by calling predict() on the model defined above
from learntools.deep_learning.decode_predictions import decode_predictions

from IPython.display import Image, display



#passing in our predictions from above

most_likely_labels = decode_predictions(preds, top=5, class_list_path='../input/resnet50/imagenet_class_index.json') #extracts the top X highest probabilities for each photo



#following prints and displays predictions and images

for i, img_path in enumerate(img_paths):

    display(Image(img_path))

    #print("Breed:", most_likely_labels[i])

    #[('id', 'breedname', 'predicted percentage'), ...]

    #to reference highest percentage breed name: most_likely_labels[i][0][1]

    predictions_dict = {}

    for unformatted_tuple in most_likely_labels[i]:

        predictions_dict[unformatted_tuple[1]] = 100 * float(unformatted_tuple[2]) #percentage in terms of 100 i.e 0.50 = 50%

    

    print("Results for " + img_path + ": ")

    print("---------- (Breed , Confidence) ----------")

    breeds = predictions_dict.keys()

    for breed in breeds:

        print(breed, ",", str(predictions_dict[breed]) + "%")

    