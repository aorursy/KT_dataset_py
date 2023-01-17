import pandas as pd

import numpy as np

import os

import shutil

import random



train_path = "../input/global-wheat-detection/train.csv"

images_path = "../input/global-wheat-detection/train/"



# We will load the csv data in a pandas dataframe and find its shape

data = pd.read_csv(train_path)

print(data.shape)



os.mkdir("un-annotated")



# We will count the number of images in the train folder

i = 0

for image in os.listdir(images_path):

    i += 1

print("Total number of images are %d", i)
data.head()
test_ids = random.sample(list(data["image_id"].unique()), 650)

test = pd.DataFrame()

for id in test_ids:

  test = test.append(data[data['image_id'] == id])

  data.drop(data[data['image_id'] == id].index, inplace = True)
test.to_csv("/kaggle/working/un-annotated/test.csv", index = False)
values = []

filename = []



# Extracting the bbox values from all the rows, removing the brackets and converting them to list and finally appending the to another list

for index, row in data.iterrows():

  values.append(row.bbox.strip('[]').split(", "))

  filename.append(row.image_id + '.jpg')



# Type converting values from string to float

# Calculating xmax and ymax by adding width and height

# Saving the value in the corresponding list

xmin = []

ymin = []

xmax = []

ymax = []

for value in values:

  xmin.append(float(value[0]))

  ymin.append(float(value[1]))

  xmax.append(float(value[2]) + float(value[0]))

  ymax.append(float(value[3]) + float(value[1]))



# Preparing anew dataframe in the required format

processed_data = {}

processed_data["filename"] = filename

processed_data["width"] = data['width']

processed_data["height"] = data['height']

data['class'] = 'wheat'

processed_data["class"] = data['class']

processed_data["xmin"] = xmin

processed_data["ymin"] = ymin

processed_data["xmax"] = xmax

processed_data["ymax"] = ymax

processed_data = pd.DataFrame(processed_data)



# Saving the newly processed data in a file

processed_data.to_csv("/kaggle/working/un-annotated/train_processed.csv", index = False)
test = pd.read_csv("/kaggle/working/un-annotated/test.csv")

values = []

filename = []



# Extracting the bbox values from all the rows, removing the brackets and converting them to list and finally appending the to another list

for index, row in data.iterrows():

  values.append(row.bbox.strip('[]').split(", "))

  filename.append(row.image_id + '.jpg')



# Type converting values from string to float

# Calculating xmax and ymax by adding width and height

# Saving the value in the corresponding list

xmin = []

ymin = []

xmax = []

ymax = []

for value in values:

  xmin.append(float(value[0]))

  ymin.append(float(value[1]))

  xmax.append(float(value[2]) + float(value[0]))

  ymax.append(float(value[3]) + float(value[1]))



# Preparing anew dataframe in the required format

processed_data = {}

processed_data["filename"] = filename

processed_data["width"] = data['width']

processed_data["height"] = data['height']

data['class'] = 'wheat'

processed_data["class"] = data['class']

processed_data["xmin"] = xmin

processed_data["ymin"] = ymin

processed_data["xmax"] = xmax

processed_data["ymax"] = ymax

processed_data = pd.DataFrame(processed_data)



# Saving the newly processed data in a file

processed_data.to_csv("/kaggle/working/un-annotated/test_processed.csv", index = False)