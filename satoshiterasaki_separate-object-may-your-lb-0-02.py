from collections import defaultdict
import os
from glob import glob 
from imageio import imread
from matplotlib import pyplot as plt
import cv2 
import pandas as pd 
import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev
import skimage
from skimage import measure 
from skimage.feature import peak_local_max
from skimage.morphology import watershed
stage1_train_path='../input/stage1_train/'
folders=os.listdir(stage1_train_path)
folders[:10]
stage1_csv_path = "../input/stage1_train_labels.csv"
df=pd.read_csv(stage1_csv_path)
df.head(5)
class CSVDecoder(object):
    def __init__(self, csv_path, train=True):
        df = pd.read_csv(csv_path)

        self.id_pixels = defaultdict(list)
        for key in set(df['ImageId']):
            subdf = df.query("ImageId=='{}'".format(key))
            value = [list(map(int, data.split(" ")))
                     for data in subdf["EncodedPixels"]]
            self.id_pixels[key] = value

        basedir='../'

        if train:
            self.stage1_data_path = os.path.join(basedir,'input','stage1_train')
        else:
            self.stage1_data_path = os.path.join(basedir,'input','stage1_test')

    def decode(self, imageid):
        img_path = os.path.join(
            self.stage1_data_path, imageid, 'images', imageid+'.png')
        img = imread(img_path)
        row, col = img.shape[:2]
        label = np.zeros(row*col, dtype=int)
        for encorded_pixels in self.id_pixels[imageid]:
            for i in range(0, len(encorded_pixels), 2):
                pos_idx = encorded_pixels[i]-1
                length = encorded_pixels[i+1]
                label[pos_idx:pos_idx+length] = 255

        label = label.reshape(col, row).transpose()
        return label

    def get_masks(self,imageid):
        img_path = os.path.join(
            self.stage1_data_path, imageid, 'images', imageid+'.png')
        img = imread(img_path)
        row, col = img.shape[:2]
        masks=[]
        for encorded_pixels in self.id_pixels[imageid]:
            mask = np.zeros(row*col, dtype=int)
            for i in range(0, len(encorded_pixels), 2):
                pos_idx = encorded_pixels[i]-1
                length = encorded_pixels[i+1]
                mask[pos_idx:pos_idx+length] = 255
            mask=mask.reshape(col,row).transpose()
            masks.append(mask)
        return masks
decoder=CSVDecoder(stage1_csv_path,train=True)
example_id = '0d3640c1f1b80f24e94cc9a5f3e1d9e8db7bf6af7d4aba920265f46cadc25e37'
decoded=decoder.decode(example_id)
plt.imshow(decoded)
image_labels = measure.label(decoded)
target_label = image_labels==4
plt.imshow(target_label)
distance = ndimage.distance_transform_edt(target_label)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=target_label)
markers = skimage.morphology.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=target_label)
plt.imshow(labels_ws)
image_labels = measure.label(decoded)
target_label = image_labels==4

contours = measure.find_contours(target_label,0.5)
contour_map=np.zeros(target_label.shape)

contours.sort(key=lambda x:-len(x))
contour=contours[0]
for c in contour:
    contour_map[int(c[0]),int(c[1])]=1

plt.imshow(contour_map)
contour=contours[0]
for c in contour:
    contour_map[int(c[0]),int(c[1])]=1
step_contour=contour.astype(int)[::5]
contour_map/=5
length=len(step_contour)
step=2
cnt=0
for i in range(0,length,step):
    cpre=step_contour[(i-step)%length]
    c = step_contour[i]
    cnxt=step_contour[(i+step)%length]
    vpre=(cpre-c)/np.linalg.norm(cpre-c)
    vnxt=(cnxt-c)/np.linalg.norm(cnxt-c)
    cos_theta=np.dot(vpre,vnxt)
    theta=np.arccos(cos_theta)
    if np.cross(vpre,vnxt) < 0 and np.rad2deg(theta)<160:
        cnt+=1
        contour_map[tuple(c)]=1 # this position c is what we call left angle corner
plt.imshow(contour_map,cmap='gray')
image_labels = measure.label(decoded)
target_label = image_labels==4
contours = measure.find_contours(target_label,0.5)
contours.sort(key=lambda x:-len(x))
contour=contours[0].astype(int)
step_contour=contour[::5]
labeled_contour_map=np.zeros(target_label.shape)

contour_label_pair=[]

for c in contour:
    nbd_value=[]
    for i in [-1,0,1]:
        for j in [-1,0,1]:
             nbd_value.append(contour_map[(c[0]+i)%target_label.shape[0],(c[1]+j)%target_label.shape[1]])

    labeled_contour_map[tuple(c)]=max(nbd_value)
    contour_label_pair.append([c, max(nbd_value)])
    
plt.imshow(labeled_contour_map,cmap='gray')
relabeled_contour_map=np.zeros(target_label.shape)
new_label=1
counter=defaultdict(int)
for c_label in contour_label_pair:
    c,label=c_label[0],c_label[1]
    if label ==1:
        new_label+=1
    counter[new_label]+=1
    relabeled_contour_map[tuple(c)]=new_label
plt.imshow(relabeled_contour_map )
kmax,vmax=0,0
for k,v in counter.items():
    if vmax < v:
        vmax=v
        kmax=k
plt.imshow(relabeled_contour_map==kmax)
labeled_contour_map=relabeled_contour_map
plt.imshow(labeled_contour_map==kmax)
img_gray=(labeled_contour_map==kmax).astype(np.uint8)*255
ret, thresh = cv2.threshold(img_gray, 127, 255,0)
_, contours,hierarchy = cv2.findContours(thresh,2,1)

displayframe=np.zeros(img_gray.shape)

for ind, cont in enumerate(contours):
    elps = cv2.fitEllipse(cont)
    pos,axis,angle=elps
    mag=0.6
    axis=(axis[0]*mag,axis[1]*mag)
    elps=(pos,axis,angle)
    #Feed elps directly into cv2.ellipse
    cv2.ellipse(displayframe,elps,(255,0,0),-1)
plt.figure(figsize=(10,5),dpi=180)
plt.subplot(1,5,1)
plt.imshow(labeled_contour_map==kmax)
plt.subplot(1,5,2)
plt.imshow(displayframe)
plt.subplot(1,5,3)
plt.imshow(target_label)
plt.subplot(1,5,4)
from copy import copy
extraced_labels_ws=copy(labels_ws)
extraced_labels_ws[~displayframe.astype(bool)]=0
buff=np.zeros(labels_ws.shape)
for i in set(extraced_labels_ws.flatten()):
    if i==0:
        continue
    tmp = labels_ws==i
    buff+=tmp
plt.imshow(buff)
plt.subplot(1,5,5)
plt.imshow(target_label-buff)

