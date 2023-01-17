# Importing required libraries

from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
from PIL import Image
def show_image(img_array, label):
    cv2.putText(img_array, label, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    plt.figure()
    plt.imshow(img_array)
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def get_image_array(folder_path):
    onlyfiles = {join(folder_path, f): f for f in listdir(folder_path) if isfile(join(folder_path, f))}
    file_paths = onlyfiles.keys()
    file_names = onlyfiles.values()
    image_array_list = list()
    image_to_show_list = list()
    file_path_list = list()
    for file_path in file_paths:
        image_to_show = np.array(Image.open(file_path))
        image_to_show = resize(image_to_show, width=400)
        image_to_show_list.append(image_to_show)
        test_image = load_img(file_path, target_size = (64, 64))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        image_array_list.append(test_image)
        # file_name_list.append(file_path)

    return image_array_list, image_to_show_list, file_names, file_paths

model_path = '../input/model-cats-dogs/image_model.h5'
image_path = '../input/catsdogs'
def load_my_model():
    classfier_load = load_model(model_path)
    path_to_images = image_path
    image_array_list, image_to_show_list, file_name_list, file_paths = get_image_array(path_to_images)
#     x = get_image_data(folder_path)
    # result = classfier_load.predict_classes(np.array(x))
    for image_array, image_to_show, file_name, file_path in zip(image_array_list, image_to_show_list, file_name_list, file_paths):
        result = classfier_load.predict(image_array)
        prediction = 'dog' if result[0][0] else 'cat'
        print(file_name, prediction)
        # print(image_path)
#         image = cv2.imread(file_path)
        # output = imutils.resize(image, width=400)
        show_image(image_to_show, prediction)

load_my_model()
print('FINISHED')