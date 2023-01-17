%matplotlib inline



# hide warnings

import warnings

warnings.simplefilter('ignore')



import os

import shutil

from pathlib import Path



import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle



from PIL import Image, ImageDraw, ImageFont

import cv2



import regex as re

import math

import random



from itertools import compress



from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import preprocess_input

from keras.utils.data_utils import GeneratorEnqueuer



from tqdm import tnrange, tqdm_notebook



from sklearn.manifold import TSNE

from sklearn.decomposition import TruncatedSVD
# download a font that can display the characters



# From https://www.google.com/get/noto/

!wget -q --show-progress https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip

!unzip -p NotoSansCJKjp-hinted.zip NotoSansCJKjp-Regular.otf > NotoSansCJKjp-Regular.otf

!rm NotoSansCJKjp-hinted.zip



# enable font for matplotlib

import matplotlib.font_manager as font_manager

path = './NotoSansCJKjp-Regular.otf'

prop = font_manager.FontProperties(fname=path)
# save and inspect input directory

INPUT = Path("../input/kuzushiji-recognition")

print(os.listdir(INPUT))



# save and inspect sub-folders of input directory

TEST = INPUT/'test_images'

TRAIN = INPUT/'train_images'

print(os.listdir(TEST)[:3])

print(os.listdir(TRAIN)[:3])



#Check the number of training and testing images

print(f"images in training dataset: {len(os.listdir(TRAIN))}")

print(f"images in test dataset: {len(os.listdir(TEST))}")
#Inspect train.csv

train_df = pd.read_csv(INPUT/'train.csv')

train_df.head()
# setup an image_id test variable

test_img_id = train_df.image_id[0]
def toPath(string):

    ''' image_id to the path to image '''

    if ".jpg" not in string:

        string = string + ".jpg"

    return string



def toID(string):

    ''' image path to image_id '''

    if string[-4:] ==".jpg":

        string = string[:-4]

    return string



print(toPath("0123"))

print(toID("0123.jpg"))
# In the training data, we get an entire string with all the characters in it and that needs to be splitted



# new list, every element is one char + all the information needed to create the bounding box

def splitEachChar(string):

    string = str(string)

    string = (re.findall(r"(?:\S*\s){5}", string))

    return [line[:-1]for line in string]



# new list, split everything by a blank

def splitEachInformation(string):

    string = str(string)

    string = string.split(" ")

    return string

        

    

    

print(splitEachChar(train_df.labels[0])[:2])

print(splitEachInformation(train_df.labels[0])[:10])
def get_unicodes(string):

    """function to get all unicode chars from a string with regex"""

    string = str(string)

    return re.findall(r'U[+][\S]*', string)
def getImageSize(image):

    """returns the image size given the image_id or path to the image"""

    path = toPath(image)

    width, height = Image.open(TRAIN/path).size

    return [width, height]



getImageSize(test_img_id)
unicode_map = {codepoint: char for codepoint, char in pd.read_csv(INPUT/'unicode_translation.csv').values}

unicode_list = list(unicode_map)



def unicodeToCharacter(unicode):

    ''' turns the unicode into the actual character '''

    return unicode_map[unicode]



# unicode to int conversion and the other way around

# unique identifier for every unicode character

def unicodeToInt(unicode):

    return unicode_list.index(unicode)



def intToUnicode(integer):

    return unicode_list[integer]



test_unicode = unicode_list[10]



print(test_unicode)

print(unicodeToCharacter(test_unicode))

print(unicodeToInt(test_unicode))

print(intToUnicode(10))
def isUnicode(string):

    ''' check whether the passed string is a unicode or not '''

    string = string.strip()

    if re.match("^U\+\w{4,5}$", string):

        return(True)

    else:

        return(False)

    



testUnicode1 = intToUnicode(10)

testUnicode2 = intToUnicode(20)



print(isUnicode(testUnicode1))

print(isUnicode(testUnicode2))

print(isUnicode(testUnicode1+"abc"))

print(isUnicode(testUnicode2+" "))
def displayImage(filepath=None, directory=None, image_id=None):

    """

    display one image with matplotlib

    

    Parameters:

    - either specify the entire filepath or (the direcory and the image_id)

    

    Returns:

    - matplotlib.pyplot figure of the image

    """

    

    if filepath == None:

        if (directory == None) and (image_id==None):

            print("path to file not specified")

            return None

        else:

            filepath=directory/toPath(image_id)

    

    plt.figure(figsize=(15,15))

    this_img = Image.open(filepath)

    plt.imshow(this_img)

    return plt



displayImage(directory=TRAIN, image_id=test_img_id)
def displayRandomImages(directory, paths=None , rows=3, columns=3):

    """

    display random images from a folder

    

    parameters:

    - directory (string or Path) of images

    - paths (string of Path) of images inside of directory

    that should be viewed. If not specified, these are

    all files inside of the directory.

    - rows (int): the number of rows that should be displayed

    - columns (int): the number of columns that should be displayed

    """

    fig = plt.figure(figsize=(20, 20))

    

    # if path is not specified, display all files in directory

    if paths == None:

        paths = os.listdir(directory)

        

    for i in range(1, rows*columns + 1):

        randomNumber = random.randint(0, len(paths)-1)

        image = Image.open(directory/paths[randomNumber])

        fig.add_subplot(rows, columns, i)

        plt.imshow(image, aspect='equal')

    plt.show()
unicode_df = pd.read_csv(INPUT/'unicode_translation.csv')

display(unicode_df.head(6))

print(len(unicode_df))
#concatenate all labels to one string

all_labels = train_df.labels.str.cat(sep=" ")



# get all unicodes in that string

all_unicodes = get_unicodes(all_labels)



# get the number of unique values from all unicodes

len(set(all_unicodes))
displayRandomImages(TRAIN)
#Check whether there are NaN columns in the training set

train_df.info()
images_nan_labels = train_df[train_df.isna().labels]['image_id'].tolist()

images_nan_paths = [str(label)+".jpg" for label in images_nan_labels]

displayRandomImages(directory=TRAIN, paths=images_nan_paths)
CHAR = Path("../input/kuzushiji-characters")
displayRandomImages(CHAR)
# take the image_id return an image with bounding boxes around each character

# image_id is the filename without the file extension (in this case .jpg)



# get all the characters and the position of the bounding boxes for an image

def getLabels(image_id):

    allLabels = train_df.loc[train_df["image_id"]==image_id].labels[0]

    allLabels = np.array(allLabels.split(" ")).reshape(-1, 5)

    return allLabels



def drawBoxAndText(ax, label):

    codepoint, x, y, w, h = label

    x, y, w, h = int(x), int(y), int(w), int(h)

    rect = Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")

    ax.add_patch(rect)

    ax.text(x+w+25, y+(h/2)+20, unicodeToCharacter(codepoint),

            fontproperties=prop,

            color="r",

           size=16)

    return ax



def displayTrainingData(image_id):

    labels = getLabels(image_id)

    plt = displayImage(directory=TRAIN, image_id=image_id)

    ax = plt.gca()



    for label in labels:

        ax = drawBoxAndText(ax, label)

        

    

displayTrainingData(test_img_id)
noOfChars = 10

noOfSamplesPerChar = 100
width = 60

height = 80
def filenameToUnicodeInt(string):

    '''

    filename to integer representing a unicode

    '''

    unicode = string.split("_")[0]

    unicodeInteger = unicodeToInt(unicode)

    return unicodeInteger
char_filenames = os.listdir(CHAR)

chars = [filenameToUnicodeInt(filename) for filename in char_filenames]



from collections import Counter

countAll = Counter(chars)



def getNmostCommonCharacters(n=10, countAll=countAll):

    """

    get a list of the most common characters in the Kuzushiji dataset

    """

    NmostCommon = countAll.most_common(n)

    NmostCommon = [unicodeID for unicodeID,frequency in NmostCommon]

    return NmostCommon
def getPathsFromCharID(char_id, noOfSamples=100):

    """

    get n(=noOfSamples) paths to chars for every char in char_id

    """

    char_id = int(char_id)

    isCharIdList = [filenameToUnicodeInt(filename)==char_id for filename in os.listdir(CHAR)]

    allCharIdPaths = list(compress(os.listdir(CHAR), isCharIdList))

    return allCharIdPaths[:noOfSamples]
paths = [getPathsFromCharID(charID, noOfSamplesPerChar) for charID in getNmostCommonCharacters(noOfChars)]

# flatten the list

paths = sum(paths, [])
# create empty images and empty labels

images = np.zeros(shape=(noOfChars * noOfSamplesPerChar, width*height), dtype=np.uint8)

labels = np.zeros(shape=(noOfChars * noOfSamplesPerChar,), dtype=np.uint8)



# flatten images + make images black and white

for index, imageName in enumerate(paths):

    filepath = str(CHAR/imageName)

    img = cv2.imread(filepath)

    img = cv2.resize(img, (width, height))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.reshape(-1, width*height)

    images[index] = img

    labels[index] = filenameToUnicodeInt(imageName)
uniqueLabels = np.array([int(i) for i in list(set(labels))], dtype=np.uint8)

uniqueUnicodeLabels = [unicodeToCharacter(intToUnicode(unicode)) for unicode in uniqueLabels]

uniqueUnicodeLabels
tsne = TSNE(n_components=2, perplexity=30)

random_train_2D = tsne.fit_transform(images)

fig = plt.figure(figsize=(10, 8))

for i in uniqueLabels:

    sns.scatterplot(random_train_2D[labels == i, 0], 

                random_train_2D[labels == i, 1], 

                label=i, s=18)

    

plt.title("Visualizating embeddings from the 10 most common Kuzushiji characters using tSNE", fontsize=16)

plt.legend(uniqueUnicodeLabels,prop=prop)

plt.show()
fig = plt.figure(figsize=(10, 8))

X_pca = TruncatedSVD(n_components=2).fit_transform(images)

for i in uniqueLabels:

    sns.scatterplot(X_pca[labels == i, 0], 

                X_pca[labels == i, 1], 

                label=i, s=18)

    

plt.title("Principal Component projection of the 10 most common Kuzushiji characters", fontsize=16)

plt.legend(uniqueUnicodeLabels,prop=prop)

plt.show()
import umap

reduce = umap.UMAP()

embedding = reduce.fit_transform(images)
fig = plt.figure(figsize=(10, 8))



for i in uniqueLabels:

    sns.scatterplot(embedding[labels == i, 0], 

                embedding[labels == i, 1], 

                label=i, s=18)



plt.title("UMAP of the 10 most common Kuzushiji characters", fontsize=16)

plt.legend(uniqueUnicodeLabels,prop=prop)

plt.show()