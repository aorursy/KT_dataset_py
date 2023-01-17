import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.io import imread, imread_collection
from skimage.filters import scharr

from sklearn.cluster import KMeans

from tqdm import tqdm

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

np.seterr(divide='ignore', invalid='ignore')
np.random.seed(131)
%matplotlib inline
TRAIN_PATH = '../input/stage1_train'
TEST_PATH = '../input/stage1_test'
TRAIN_IMAGE_PATTERN = "%s/{}/images/{}.png" % TRAIN_PATH
TRAIN_MASK_PATTERN = "%s/{}/masks/*.png" % TRAIN_PATH
TEST_IMAGE_PATTERN = "%s/{}/images/{}.png" % TEST_PATH
def read_img(img_id, flag_train=True):
    if flag_train:
        img_path = TRAIN_IMAGE_PATTERN.format(img_id, img_id)
    else:
        img_path = TEST_IMAGE_PATTERN.format(img_id, img_id)
    img = imread(img_path)
    img = img[:, :, :3]
    return img
def masks_reader(mask_id):
    mask_path = TRAIN_MASK_PATTERN.format(mask_id, mask_id)
    masks = imread_collection(mask_path).concatenate()
    return masks
def get_spector(ids):
    img = read_img(ids)
    masks = masks_reader(ids)
    spector = [[],[],[]]
    for m in masks:
        i = img.copy()
        ir, ig, ib = i[:,:,0], i[:,:,1], i[:,:,2]
        ir, ig, ib, m = ir.flatten(), ig.flatten(), ib.flatten(), m.flatten()
        ir, ig, ib = ir[m>0], ig[m>0], ib[m>0]
        spector[0].extend(ir.tolist())
        spector[1].extend(ig.tolist())
        spector[2].extend(ib.tolist())
    return spector
def read_concat_mask(mask_id):
    masks = masks_reader(mask_id)
    _,height, width = masks.shape
    num_masks = masks.shape[0]
    mask = np.zeros((height, width), np.uint32)
    for index in range(0, num_masks):
        mask[masks[index] > 0] = 1
    return mask
def read_contur_mask(mask_id):
    masks = masks_reader(mask_id)
    _,height, width = masks.shape
    num_masks = masks.shape[0]
    mask = np.zeros((height, width), np.uint32)
    c = 125
    for index in range(0, num_masks):
        dm = scharr(masks[index])
        mask[dm > 0] = c
        if c!=255:
            c += 1
        else:
            c = 125
    return mask
def image_ids(root_dir, ignore=[]):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids
def visualizer(imgs, n=4, figsize=(16,16), title=''):
    fig = plt.figure(figsize=figsize)

    n_samples = list(range(len(imgs)))

    for i in range(int(n**2)):
        try:
            rsample = random.choice(n_samples)
            n_samples.remove(rsample)
            img = imgs[rsample]
            ax = fig.add_subplot(n,n,i+1)
            ax.imshow(img)
            ax.axis('off')
        except IndexError:
            pass
    fig.suptitle(title)
def contur_visualizer(imgs, contur, n=2, figsize=(16,16), title=''):
    fig = plt.figure(figsize=figsize)

    n_samples = list(range(len(imgs)))

    for i in range(int(n**2)):
        try:
            rsample = random.choice(n_samples)
            n_samples.remove(rsample)
            img = imgs[rsample]
            cont = contur[rsample]
            ax = fig.add_subplot(n,n,i+1)
            ax.imshow(img)
            ax.imshow(cont, alpha=0.5)
            ax.axis('off')
        except IndexError:
            pass
    fig.suptitle(title)
    plt.savefig('contur.png')
train_img_ids = image_ids(TRAIN_PATH)
test_img_ids = image_ids(TEST_PATH)
train_image = [read_img(i) for i in tqdm(train_img_ids, desc='Reading train image')]
test_image = [read_img(i, flag_train=False) for i in tqdm(test_img_ids, desc='Reading test image')]
visualizer(train_image)
def get_color_state1(imgs):
    color_state = []
    for img in imgs:
        g = np.mean(img[:,:,0])
        grm = np.mean(img[:,:,1]-img[:,:,0])
        grs = np.std(img[:,:,1]-img[:,:,0])
        color_state.append([g,grm, grs])
    return color_state    
train_cs1 = get_color_state1(train_image)
test_cs1 = get_color_state1(test_image)
X_tr = train_cs1
X_te = test_cs1

kmeans = KMeans(n_clusters=4).fit(X_tr)
train_cl1 = np.argmin(kmeans.transform(X_tr), -1)
test_cl1 = np.argmin(kmeans.transform(X_te), -1)
for j in range(4):
    train_img_cl = []
    for i in range(len(train_cl1)):
        if train_cl1[i]==j:
            train_img_cl.append(train_image[i])
    visualizer(train_img_cl, title='Cluster '+str(j))
for j in range(4):
    test_img_cl = []
    for i in range(len(test_cl1)):
        if test_cl1[i]==j:
            test_img_cl.append(test_image[i])
    visualizer(test_img_cl, title='Cluster '+str(j))
utest_cl1 = np.unique(test_cl1)
freez_cl1 = set([0])&set(utest_cl1)
utest_cl1
train_image1 = []
train_ids1 = []
train_clout = {
    'ids':[],
    'image':[],
    'cluster':[],
}
for i in range(len(train_cl1)):
    if train_cl1[i] in (set(utest_cl1)^set(freez_cl1)):
        train_image1.append(train_image[i])
        train_ids1.append(train_img_ids[i])
    elif train_cl1[i] in freez_cl1:
        train_clout['ids'].append(train_img_ids[i])
        train_clout['image'].append(train_image[i])
        train_clout['cluster'].append(train_cl1[i])
test_image1 = []
test_ids1 = []
test_clout = {
    'ids':[],
    'image':[],
    'cluster':[],
}
for i in range(len(test_cl1)):
    if test_cl1[i] in (set(utest_cl1)^set(freez_cl1)):
        test_image1.append(test_image[i])
        test_ids1.append(test_img_ids[i])
    elif test_cl1[i] in freez_cl1:
        test_clout['ids'].append(test_img_ids[i])
        test_clout['image'].append(test_image[i])
        test_clout['cluster'].append(test_cl1[i])
def get_color_state2(imgs):
    color_state = []
    for img in imgs:
        r = np.mean(img[:,:,0])
        b = np.mean(img[:,:,2])

        rs = np.std(img[:,:,0])
        bs = np.std(img[:,:,2])
        color_state.append([r, rs, b, bs])
    return color_state  
train_cs2 = get_color_state2(train_image1)
test_cs2 = get_color_state2(test_image1)
X_tr = train_cs2
X_te = test_cs2

kmeans = KMeans(n_clusters=2).fit(X_tr)
train_cl2 = np.argmin(kmeans.transform(X_tr), -1)
test_cl2 = np.argmin(kmeans.transform(X_te), -1)
utest_cl2 = np.unique(test_cl2)
freez_cl2 = set([0])&set(utest_cl2)
utest_cl2
for i in range(len(train_cl2)):
    if train_cl2[i] in freez_cl2:
        train_clout['ids'].append(train_ids1[i])
        train_clout['image'].append(train_image1[i])
        train_clout['cluster'].append(train_cl2[i]+1)
for i in range(len(test_cl2)):
    if test_cl2[i] in freez_cl2:
        test_clout['ids'].append(test_ids1[i])
        test_clout['image'].append(test_image1[i])
        test_clout['cluster'].append(test_cl2[i]+1)
for j in range(2):
    train_img_cl = []
    for i in range(len(train_clout['cluster'])):
        if train_clout['cluster'][i]==j:
            train_img_cl.append(train_clout['image'][i])
    visualizer(train_img_cl, title='Cluster '+str(j))
for j in range(2):
    test_img_cl = []
    for i in range(len(test_clout['cluster'])):
        if test_clout['cluster'][i]==j:
            test_img_cl.append(test_clout['image'][i])
    visualizer(test_img_cl, title='Cluster '+str(j))
spector = {
    0:[],
    1: [[],[],[]],
}
for i, c in tqdm(zip(train_clout['ids'], train_clout['cluster']), desc='Get Spetor'):
    if c==0:
        spector[c].extend(get_spector(i)[0])
    else:
        spector[c][0].extend(get_spector(i)[0])
        spector[c][1].extend(get_spector(i)[1])
        spector[c][2].extend(get_spector(i)[2])
sns.distplot(spector[0]);
sns.distplot(spector[1][0], color='red', label='r')
sns.distplot(spector[1][1], color='green', label='g')
sns.distplot(spector[1][2], color='blue', label='b')
plt.legend();
train_clout['contur'] = []
for i in tqdm(train_clout['ids'], desc='Add Masks'):
    train_clout['contur'].append(read_contur_mask(i))
contur_visualizer(train_clout['image'], train_clout['contur'])