import sys

package_dir = '../input/autokeras/autokeras'

sys.path.insert(0, package_dir)
import autokeras as ak

from autokeras.utils import pickle_from_file
autokeras_model = pickle_from_file("../input/autokeras-model-v3/autokeras_model_v3.h5")
"""

# you can also convert the model to Keras but pre-trained weights disappear in this case

autokeras_model.export_keras_model('keras_model.h5')



from keras.models import load_model

keras_model = load_model('keras_model.h5')

keras_model.summary()

"""
import numpy as np

import pandas as pd

import cv2

from tqdm import tqdm
IMG_SIZE = 224
df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
df.head()
#https://www.kaggle.com/taindow/pre-processing-train-and-test-images



def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = gray_img>tol        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0):

            return img

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img



def circle_crop_v2(img):

    img = cv2.imread(img)

    img = crop_image_from_gray(img)



    height, width, depth = img.shape

    largest_side = np.max((height, width))

    img = cv2.resize(img, (largest_side, largest_side))



    height, width, depth = img.shape



    x = int(width / 2)

    y = int(height / 2)

    r = np.amin((x, y))



    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)



    return img



def getImagePixelsNew(img_name):

    

    image = circle_crop_v2('../input/aptos2019-blindness-detection/test_images/%s.png' % img_name)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    

    #normalize in scale of 0, +1

    img_pixels = image / 255 #0 to 254

    

    return img_pixels
df['pixels'] = df['id_code'].apply(getImagePixelsNew)
#df.head()
df.iloc[0]['pixels'].shape
features = []



pbar = tqdm(range(0, df.shape[0]), desc='Processing')

for index in pbar:

    features.append(df.iloc[index]['pixels'])



print("features variable created: ",len(features))
predictions = []



pbar = tqdm(range(0, len(features)), desc='Processing')



for index in pbar:

    prediction = autokeras_model.predict(np.expand_dims(features[index], axis = 0))

    predictions.append(prediction[0])



#predictions = autokeras_model.predict(features)
predictions[0:10]
df['diagnosis'] = predictions

df = df.drop(columns=['pixels'])
df.head()
df.to_csv("submission.csv", index=False)