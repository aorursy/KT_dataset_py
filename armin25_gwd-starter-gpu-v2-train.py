!pip install -qq segmentation-models
import os

import sys

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from skimage.io import imread

from skimage.transform import resize

from PIL import Image, ImageDraw

from tqdm.notebook import tqdm

import segmentation_models as sm
EPOCHS = 10



IMG_WIDTH = 256

IMG_HEIGHT = 256

IMG_CHANNELS = 3

TRAIN_PATH = '/kaggle/input/global-wheat-detection/train/'
PATH = "../input/global-wheat-detection/"

train_folder = os.path.join(PATH, "train")

train_csv_path = os.path.join(PATH, "train.csv")

df = pd.read_csv(train_csv_path)



df.head()
train_ids = os.listdir(TRAIN_PATH)

len(train_ids)
def make_polygon(coords):

    xm, ym, w, h = coords

    xm, ym, w, h = xm / 4, ym / 4, w / 4, h / 4

    polygon = [(xm, ym), (xm, ym + h), (xm + w, ym + h), (xm + w, ym)]

    return polygon



masks = dict() # dictionnary containing all masks



for img_id, gp in tqdm(df.groupby("image_id")):

    gp['polygons'] = gp['bbox'].apply(eval).apply(lambda x: make_polygon(x))



    img = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 0)

    for pol in gp['polygons'].values:

        ImageDraw.Draw(img).polygon(pol, outline=1, fill=1)



    mask = np.array(img, dtype=np.uint8)

    masks[img_id] = mask
# Get and resize train images and masks

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Getting and resizing train images and masks ... ')

sys.stdout.flush()



for n, id_ in tqdm(enumerate(train_ids[:]), total=len(train_ids)):

    path = TRAIN_PATH + id_

    img = imread(path)[:,:,:IMG_CHANNELS]

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    

    id_clean = id_.split('.')[0]

    if id_clean in masks.keys():

        Y_train[n] = masks[id_clean][:, :, np.newaxis]
X_train.shape, Y_train.shape
a=2

plt.imshow(X_train[a])
plt.imshow(Y_train[a,:,:,0].astype(int))
model = sm.Unet('efficientnetb3', encoder_weights='imagenet')
model.compile(

    'Adam',

    loss=sm.losses.bce_jaccard_loss,

    metrics=[sm.metrics.iou_score],

)
model.fit(

   x=X_train,

   y=Y_train,

   batch_size=16,

   epochs=EPOCHS

)
model.save('model.h5')