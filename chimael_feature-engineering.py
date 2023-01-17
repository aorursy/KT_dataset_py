import numpy as np
import pandas as pd
import skimage.io
import matplotlib.pyplot as plt
#import training data
train = pd.read_csv("../input/train.csv")
train.head()
path_to_train = '../input/train/'
def load_image(file):
    image_red_ch = skimage.io.imread(path_to_train+file+'_red.png')
    image_yellow_ch = skimage.io.imread(path_to_train+file+'_yellow.png')
    image_green_ch = skimage.io.imread(path_to_train+file+'_green.png')
    image_blue_ch = skimage.io.imread(path_to_train+file+'_blue.png')
    image = np.stack((image_green_ch, image_red_ch, image_blue_ch, image_yellow_ch))
    return image
from skimage.filters import threshold_otsu
def threshod_image(img):
    bw_img = np.zeros_like(img, dtype=bool)
    for i, arr in enumerate(img):
        bw_img[i] = arr > threshold_otsu(arr)
    return bw_img
def mask_green(bw_img):
    mask_img_red = bw_img[0] & bw_img[1]
    mask_img_blue = bw_img[0] & bw_img[2]
    mask_img_yellow = bw_img[0] & bw_img[3]
    return np.stack((bw_img[0], mask_img_red, mask_img_blue, mask_img_yellow))
def compute_ratios(mask_img):
    ratios = []
    for i in range(1,mask_img.shape[0]):
        ratios.append(mask_img[i].sum()/mask_img[0].sum())
    return ratios
def transform(file):
    a = load_image(file)
    bw_img = threshod_image(a)
    mask_img = mask_green(bw_img)
    return compute_ratios(mask_img)
from sklearn.preprocessing import MultiLabelBinarizer
def dataset(size=100):
    targets = []
    features = []
    c = 0
    for i, row in train.sample(size, random_state=1).iterrows():
        c+=1
        targets.append([int(x) for x in row[1].split(' ')])
        features.append(transform(row[0]))
        if c % 10 == 0:
            print("Processing %.2f" % ((c*100)/size), end='\r')
    return np.array(features), MultiLabelBinarizer().fit_transform(targets)
features, targets = dataset(1000)
features.shape
targets.shape
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(features[:700], targets[:700]) 
preds = neigh.predict(features[700:])
f1_score(targets[700:], preds, average='macro')
