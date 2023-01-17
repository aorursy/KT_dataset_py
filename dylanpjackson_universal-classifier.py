img_path = '../input/images/hedgehog.jpg'
import numpy as np

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



image_size = 224 



"""

    Cleans up image before testing

"""

def pre_process_image(img_path, img_height=image_size, img_width=image_size):

    img = [load_img(img_path, target_size=(img_height, img_width))]

    img_array = np.array([img_to_array(img[0])])

    output = preprocess_input(img_array)

    return output
from tensorflow.python.keras.applications import ResNet50



model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

test_data = pre_process_image(img_path)

predictions = model.predict(test_data)
from learntools.deep_learning.decode_predictions import decode_predictions

from IPython.display import Image, display



labels = decode_predictions(predictions, top=4, class_list_path='../input/resnet50/imagenet_class_index.json')



display(Image(img_path))



guess = labels[0][0][1]

percent = float(labels[0][0][2]) * 100



print("Results for " + img_path + ": ")

print(guess + ": " + str(percent) + " %")