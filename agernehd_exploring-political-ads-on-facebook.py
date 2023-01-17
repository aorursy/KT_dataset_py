# Data processing 
import numpy as np
import pandas as pd
import random

# Visulalization
import matplotlib.pyplot as plt
from IPython.core import display as ICD
from torchvision import transforms
import imageio, PIL, cv2

# utilities
import os
from glob import glob
import shutil
import zipfile


# Load data
path = "../input/russian political influence campaigns/Russian Political Influence Campaigns/Supplementary Data/From Russian Ad Explorer/images.zip"
with zipfile.ZipFile(path, "r") as f:
    f.extractall('.')

# Read CSV
data_path = '../input/russian political influence campaigns/Russian Political Influence Campaigns/Supplementary Data/From Data World/FacebookAds.csv'
df = pd.read_csv(data_path)
df.head()
# Summary stats
df.describe()
# Averages
click_count_per_add = df.Clicks.sum()/df.Clicks.count()
click_ratio = (df.AdSpend.sum()/63)/df.Clicks.count()
ad_spend_count_ratio = (df.AdSpend.sum()/63)/df.Clicks.count()
print('Total number of Ads purcheased by IRA from 2015 to 2017: ', df.Clicks.count())
print('Total dollar spent by IRA from 2015 to 2017: ${:.2f}'.format(df.AdSpend.sum()/63))
print('Average cost per Ad: ${:.2f}'.format(ad_spend_count_ratio))
print('Average number of clicks per Ad: {:.2f}'.format(click_count_per_add))
print('Avg cost per click: ${:.2f}'.format(click_ratio))
# def function to load and normalize images
def load_image(path, size):
    img = PIL.Image.open(path)
    normalise_img = transforms.ToTensor()
    img_tensor = normalise_img(img).unsqueeze(0)
    img_np = img_tensor.numpy()
    return img, img_tensor, img_np

input_image = 'images/2016-01/P10001365.-000.png'
input_img, input_tensor, input_np = load_image(input_image, size=[1024, 1024])
input_img
# trgeting different community 
inputImage = 'images/2015-06/P10002571.-001.png'
input_img, input_tensor, input_np = load_image(inputImage, size=[1024, 1024])
input_img
multiple_images = glob('images/2015-06/**')

def plot_images(path, begin, end):
    i = 0
    plt.rcParams['figure.figsize'] = (25.0, 25.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    for a in multiple_images[begin:end]:
        im = cv2.imread(a)
        im = cv2.resize(im, (1024, 1024))
        plt.subplot(3, 3, i+1)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        i +=1
    plt.show()
plot_images(multiple_images, 0, 9)
plot_images(multiple_images, 9, 18)
plot_images(multiple_images, 18, 27)
print('Number of add targeted specific groups\n')
group_count = df.FriendsOfConnections.value_counts()
group_count2 = df.PeopleWhoMatch.value_counts()
print(group_count.head(5), '\n\n', group_count2.head(5), '\n')
path = 'images'
shutil.rmtree(path)