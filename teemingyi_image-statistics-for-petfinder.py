from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 

from IPython.core.display import HTML 
from IPython.display import Image

images_path = '../input/train_images/'
imgs = os.listdir(images_path)

features = pd.DataFrame()
features['image'] = imgs
features = features.loc[['-1.' in x for x in features.image]]
images_path = '../input/test_images/'
imgs = os.listdir(images_path)

features_test = pd.DataFrame()
features_test['image'] = imgs
features_test = features_test.loc[['-1.' in x for x in features_test.image]]
def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 1000
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent
def perform_color_analysis(img):

    path = images_path + img 
    im = IMG.open(path) #.convert("RGB")
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        light_percent1, dark_percent1 = -1, -1
        light_percent2, dark_percent2 = -1, -1

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    return dark_percent, light_percent
from tqdm import tqdm
tqdm.pandas()
import time
images_path='../input/train_images/'
start=time.time()
features['dullness_whiteness'] = features['image'].apply(lambda x : perform_color_analysis(x))
print(time.time()-start)
features['dullness'] = features.dullness_whiteness.map(lambda x: x[0])
features['whiteness'] = features.dullness_whiteness.map(lambda x: x[1])
topdull = features.sort_values('dullness', ascending = False)
for j,x in topdull.head(5).iterrows():
    
    path = images_path + x['image']
    html = "<h4>Image : "+x['image']+" &nbsp;&nbsp;&nbsp; (Dullness : " + str(x['dullness']) +")</h4>"
    display(HTML(html))
    display(IMG.open(path).resize((300,300), IMG.ANTIALIAS))
topbright = features.sort_values('whiteness', ascending = False)
for j,x in topbright.head(5).iterrows():
    images_path='../input/train_images/'
    path = images_path + x['image']
    html = "<h4>Image : "+x['image']+" &nbsp;&nbsp;&nbsp; (Dullness : " + str(x['dullness']) +")</h4>"
    display(HTML(html))
    display(IMG.open(path).resize((300,300), IMG.ANTIALIAS))
def average_pixel_width(img):
    path = images_path + img 
    im = IMG.open(path)    
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100
features['average_pixel_width'] = features['image'].apply(average_pixel_width)
tempdf = features.sort_values('average_pixel_width').head()
tempdf
for j,x in tempdf.head(6).iterrows():
    path = images_path + x['image']
    html = "<h4>Image : "+x['image']+" &nbsp;&nbsp;&nbsp; (Average Pixel Width : " + str(x['average_pixel_width']) +")</h4>"
    display(HTML(html))
    display(IMG.open(path).resize((300,300), IMG.ANTIALIAS))
def getSize(filename):
    filename = images_path + filename
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    filename = images_path + filename
    img_size = IMG.open(filename).size
    return img_size 
features['image_size'] = features['image'].apply(getSize)
features['temp_size'] = features['image'].apply(getDimensions)
features['width'] = features['temp_size'].apply(lambda x : x[0])
features['height'] = features['temp_size'].apply(lambda x : x[1])

features.head()
def get_blurrness_score(image):
    path =  images_path + image 
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm
features['blurrness'] = features['image'].apply(get_blurrness_score)
tempdf = features.sort_values('blurrness')
for y,x in tempdf.head(5).iterrows():
    path = images_path + x['image']
    html = "<h4>Image : "+x['image']+" &nbsp;&nbsp;&nbsp; (Blurrness : " + str(x['blurrness']) +")</h4>"
    display(HTML(html))
    display(IMG.open(path).resize((300,300), IMG.ANTIALIAS))
images_path='../input/test_images/'
start=time.time()
features_test['dullness_whiteness'] = features_test['image'].apply(lambda x : perform_color_analysis(x))
print(time.time()-start)
features_test['dullness'] = features_test.dullness_whiteness.map(lambda x: x[0])
features_test['whiteness'] = features_test.dullness_whiteness.map(lambda x: x[1])
features_test['average_pixel_width'] = features_test['image'].apply(average_pixel_width)
features_test['image_size'] = features_test['image'].apply(getSize)
features_test['temp_size'] = features_test['image'].apply(getDimensions)
features_test['width'] = features_test['temp_size'].apply(lambda x : x[0])
features_test['height'] = features_test['temp_size'].apply(lambda x : x[1])
features_test['blurrness'] = features_test['image'].apply(get_blurrness_score)
features.to_csv('train_image.csv',index=False)
features_test.to_csv('test_image.csv',index=False)