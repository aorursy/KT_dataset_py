import os

for dirname, _, filenames in os.walk('/kaggle/input'):

        print(dirname)
import numpy as np

import pandas as pd

import cv2

from tqdm import tqdm_notebook as tqdm
def get_image_names(dataframe) : 

    image_names = dataframe["image_name"].values

    image_names = image_names + ".jpg"

    return image_names
def get_info(image_names) : 

    image_names = np.array(image_names)

    

    print("Length = ", len(image_names))

    print("Type = ", type(image_names))

    print("Shape = ", image_names.shape)

    

    return image_names
from scipy.stats import skew



def extract_information(image_names, directory) : 

    image_statistics = pd.DataFrame(index = np.arange(len(image_names)),

                                    columns = ["image_name", "path", "rows", "columns", "channels", 

                                              "image_mean", "image_standard_deviation", "image_skewness",

                                              "mean_red_value", "mean_green_value", "mean_blue_value"])

    i = 0 

    for name in tqdm(image_names) : 

        path = os.path.join(directory, name)

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        image_statistics.iloc[i]["image_name"] = name

        image_statistics.iloc[i]["path"] = path

        image_statistics.iloc[i]["rows"] = image.shape[0]

        image_statistics.iloc[i]["columns"] = image.shape[1]

        image_statistics.iloc[i]["channels"] = image.shape[2]

        image_statistics.iloc[i]["image_mean"] = np.mean(image.flatten())

        image_statistics.iloc[i]["image_standard_deviation"] = np.std(image.flatten())

        image_statistics.iloc[i]["image_skewness"] = skew(image.flatten())

        image_statistics.iloc[i]["mean_red_value"] = np.mean(image[:,:,0])

        image_statistics.iloc[i]["mean_green_value"] = np.mean(image[:,:,1])

        image_statistics.iloc[i]["mean_blue_value"] = np.mean(image[:,:,2])

        

        i = i + 1

        del image

        

    return image_statistics
train_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"

train = pd.DataFrame(pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv"))

train.head()
image_names = get_image_names(train)

image_names = get_info(image_names)
#image_statistics = extract_information(image_names[0:5000], train_dir) # repeat this for image_names[5000:10k], image_names[10k-15k]...so on till 33126

#image_statistics.to_csv("melanoma_image_statistics_compiled_01", index = False)# save each one. I have computed it all beforehand, so wrote only one for instance.   
test_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/test/"

test = pd.DataFrame(pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv"))

test.head()
image_names = get_image_names(test)

image_names = get_info(image_names)
#image_statistics = extract_information(image_names[5000:10982], test_dir)

#image_statistics.to_csv("melanoma_image_statistics_compiled_test_02", index = False)   