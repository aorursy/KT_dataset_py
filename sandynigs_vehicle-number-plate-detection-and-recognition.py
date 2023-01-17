# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import urllib3

import matplotlib.pyplot as plt

import cv2

import time

from tqdm import tqdm

from PIL import Image

# PIL is python imaging library.

from urllib.request import urlopen

import pytesseract





# Libraries fro DL. 

from keras.applications.vgg16 import VGG16

from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout

from keras.models import Model, Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam
df = pd.read_json("/kaggle/input/vehicle-number-plate-detection/Indian_Number_plates.json", lines=True)

df.head()



print(df.shape)

# Make a seprate directory to save number plates.

# You have to run this operation only once.



os.mkdir("Indian Number Plates")

# In dataset dictionary we will capture specific properties of every image.

# Have run this portion in local.



dataset = dict()

dataset["image_name"] = list()

dataset["image_width"] = list()

dataset["image_height"] = list()

dataset["top_x"] = list()

dataset["top_y"] = list()

dataset["bottom_x"] = list()

dataset["bottom_y"] = list()

#Code was not working here , hence I ran it in local



counter = 0



for index, row in tqdm(df.iterrows()):

    # Iterate over DataFrame rows as (index, Series) pairs.

    # Here series is a pandas series.

    # print("Trying to fetch image: ",row["content"])

    res = urlopen(row["content"])

    # This line here is trying to access the url being pointed to in each row, where iamge is saved.

    

    img = Image.open(res)

    img = img.convert('RGB')

    # Returns a converted copy of this image.

    

    img.save("Indian Number Plates/licensed_car{}.jpeg".format(counter), "JPEG")

    # Saves the image under the given filename.

    

    # Create a dataset for all the images with properties.

    dataset["image_name"].append("licensed_car{}".format(counter))



    data = row["annotation"]

    

    dataset["image_width"].append(data[0]["imageWidth"])

    dataset["image_height"].append(data[0]["imageHeight"])

    dataset["top_x"].append(data[0]["points"][0]["x"])

    dataset["top_y"].append(data[0]["points"][0]["y"])

    dataset["bottom_x"].append(data[0]["points"][1]["x"])

    dataset["bottom_y"].append(data[0]["points"][1]["y"])

    

    counter += 1

print("Downloaded {} car images.".format(counter))



df_changed = pd.DataFrame(dataset)



df_changed.to_csv("ConvertedImagesProperties.csv", index=False)

license_plate_df = pd.read_csv("ConvertedImagesProperties.csv")

license_plate_df["image_name"] = license_plate_df["image_name"] + ".jpeg"

license_plate_df.drop(["image_width", "image_height"], axis=1, inplace=True)

license_plate_df.head()
random_test_samples = np.random.randint(0, len(df), 5)

reduced_df = license_plate_df.drop(random_test_samples, axis=0)
WIDTH = 224

HEIGHT = 224

CHANNEL = 3



def get_number(index):

    image = cv2.imread("Indian Number Plates/" + license_plate_df["image_name"].iloc[index])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, dsize=(WIDTH, HEIGHT))



    tx = int(license_plate_df["top_x"].iloc[index] * WIDTH)

    ty = int(license_plate_df["top_y"].iloc[index] * HEIGHT)

    bx = int(license_plate_df["bottom_x"].iloc[index] * WIDTH)

    by = int(license_plate_df["bottom_y"].iloc[index] * HEIGHT)

    print("Original image with rectangle boundary->")

    image_with_boundary = cv2.rectangle(image, (tx, ty), (bx, by), (0, 0, 255), 1)

    plt.imshow(image_with_boundary)

    plt.show()



    #Crop the image

    im2 = image.copy()

    im2_crop = im2[ty:by, tx:bx]

    print("Cropped image ->")

    plt.imshow(im2_crop)

    plt.show()



    

    #Preprocessing of image

    #Converting to grayscale

    gray = cv2.cvtColor(im2_crop, cv2.COLOR_BGR2GRAY) 

    #perform thresholding

    ret,thresh1 = cv2.threshold(np.array(gray), 125, 255, cv2.THRESH_BINARY)

    

    #Applying tesseract

    custom_config = r'--oem 3 --psm 6'

    string_num = pytesseract.image_to_string(thresh1, config=custom_config)

    if(len(string_num) == 0):

        print("Can not read image")

        return 1

    else:

        print(string_num)

        return 0
get_number(4)
miss_count = 0

for i in range (0,len(license_plate_df)):

    print("Figure {}".format(i))

    miss_count += get_number(i)

    print("------------------")

print("Total images miss {}".format(miss_count))