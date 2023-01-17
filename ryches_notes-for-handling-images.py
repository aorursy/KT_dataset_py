#I don't think this line needs explaining
import pandas as pd
#load the train csv and grab the images
images = pd.read_csv("../input/train.csv")[["image"]]
#let's measure some time
import time
#I'm sure importing open is probably a terrible pythonic sin, but yolo
from PIL.Image import open as PILRead
start = time.time()
for image in images[0:1000]:
        img = PILRead('./data/competition_files/train_jpg/' + str(image) + ".jpg")
print(time.time() - start)
i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = PILRead('./data/competition_files/train_jpg/' + str(image) + ".jpg")
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
from cv2 import imread as cvimread
start = time.time()
for image in X["image"][0:1000]:
    img = cvimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
print(time.time() - start)
start = time.time()
for image in X["image"][0:1000]:
    img = cvimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
    print(img.shape)
print(time.time() - start)
type(img)
i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = PILRead('./data/competition_files/train_jpg/' + str(image) + ".jpg")
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)
from keras.preprocessing.image import img_to_array
i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = img_to_array(PILRead('./data/competition_files/train_jpg/' + str(image) + ".jpg"))
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)
from scipy.misc import imread as scimread
i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = (scimread('./data/competition_files/train_jpg/' + str(image) + ".jpg"))
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)
from imageio import imread as ioimread

i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = ioimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)
from matplotlib.image import imread as matimread
i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = (matimread('./data/competition_files/train_jpg/' + str(image) + ".jpg"))
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)
from keras.preprocessing.image import load_img
i = 0
start = time.time()
for image in X["image"][0:1000]:
    try:
        img = img_to_array(load_img('./data/competition_files/train_jpg/' + str(image) + ".jpg"))
        i += 1
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start, i/1000)
type(img)
start = time.time()
for image in X["image"][0:1000]:
    img = cvimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
    try:
        img.shape
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start)
from cv2 import resize
start = time.time()
for image in X["image"][0:1000]:
    img = cvimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
    try:
        img = resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        img.shape
    except:
        img = np.zeros(shape = (224, 224, 3))
print(time.time() - start)

import concurrent.futures
def resize_img(image):
    img = cvimread('./data/competition_files/train_jpg/' + str(image) + ".jpg")
    try:
        img = resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        img.shape
    except:
        img = np.zeros(shape = (224, 224, 3))
    return img
start = time.time()
resized_imgs = []
with concurrent.futures.ThreadPoolExecutor(max_workers = 16) as executor:
    for value in executor.map(resize_img, X["image"][0:1000]):
        resized_imgs.append(value)
print(time.time() - start)      
import concurrent.futures
def resize_img(image):
    img = cvimread('./data/competition_files/train_jpg_resized/' + str(image) + ".jpg")
    try:
        img.shape
    except:
        img = np.zeros(shape = (224, 224, 3))
    return img
start = time.time()
resized_imgs = []
with concurrent.futures.ThreadPoolExecutor(max_workers = 16) as executor:
    for value in executor.map(resize_img, X["image"][0:1000]):
        resized_imgs.append(value)
print(time.time() - start)    