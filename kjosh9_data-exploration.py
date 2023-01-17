import pandas as pd
train_data = pd.read_csv("../input/train.csv")
train_data.head()
whale_data = train_data.groupby("Id").Image.nunique()
whale_data
pd.value_counts(whale_data).plot.bar(x='Pictures', y='Whales')
train_data.shape
no_of_new_whales = 9664
size_of_training_data = 25361

percentage_of_images_are_new_whales = no_of_new_whales / size_of_training_data
percentage_of_images_are_new_whales
whale_data.sort_values(ascending=False).head().plot.bar()
whale_data = whale_data.drop("new_whale")
whale_data.sort_values(ascending=False).head().plot.bar()
whale_data.sort_values(ascending=False).iloc[0:30].plot.bar()
whale_data.sort_values(ascending=False).iloc[31:60].plot.bar()
whale_data.sort_values(ascending=False).iloc[61:90].plot.bar()
whale_data.sort_values(ascending=False).iloc[91:200].plot.bar()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sample_image = mpimg.imread('../input/train/0a0c1df99.jpg')
imgplot = plt.imshow(sample_image)
plt.show()
import os
import cv2

largest_width = 0
largest_height = 0
channels_list = []
for filename in os.listdir('../input/train/'):
    img = cv2.imread('../input/train/' + filename)
    height, width, channels = img.shape
    if channels not in channels_list:
        channels_list.append(channels)
    if width > largest_width:
        largest_width = width
    if height > largest_height:
        largest_height = height
        
print("Largest width: ", width)
print("Largest height: ", height)
print("Channels present: ", channels_list)