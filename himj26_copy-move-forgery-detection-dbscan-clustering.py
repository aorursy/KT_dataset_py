!pip uninstall opencv-python -y
#downgrade OpenCV a bit since some none-free features are not avilable
!pip install opencv-contrib-python==3.4.2.17 --force-reinstall
#import libraries 
import os
import cv2
import matplotlib.pyplot as plt
import re
from sklearn.cluster import DBSCAN  # For DBSCAN
import numpy as np
%matplotlib inline

image_paths=[] #List to store path of all images

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if '.txt' in filename:
            continue
        image_paths.append(os.path.join(dirname, filename))
original_images=[]
tampered_images=[]

for path in image_paths:
    
    if 'tamp' in path:              # As Observed from the above list tampered images name has tamp
        tampered_images.append(path)
    else:
        original_images.append(path)
tampered_images.sort()
original_images.sort()
print(len(original_images),len(tampered_images))
def plot_image(img,size=(8,8)):
    plt.figure(figsize = size)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) #Since opencv store images as BGR

def siftDetector(img):
    sift = cv2.xfeatures2d.SIFT_create()
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    key_points, descriptors = sift.detectAndCompute(gray, None)
    return key_points,descriptors

def get_original(tampered):
    name=re.findall(r'.*/(.*)tamp.*',tampered)
    original_index=-1
    if len(name)<1:
        return -1
    for index,names in enumerate(original_images):
        if name[0] in names:
            original_index=index
            break
            
    if original_index==-1:
        return original_index,-1
    else:
        image=cv2.imread(original_images[original_index])
        return image,original_index

def show_sift_features(color_img, kp,size=(8,8)):
    gray_img=cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)
    plt.figure(figsize = size)
    plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))
tampered1=cv2.imread(tampered_images[0])
plot_image(tampered1)
original1 , index=get_original(tampered_images[0])
if index!=-1:
    plot_image(original1)
def make_clusters(de,eps=40,min_sample=2):
    clustering = DBSCAN(eps=eps, min_samples=min_sample).fit(de)
    return clustering

def locate_forgery(img,clustering,kps):
    forgery=img.copy()
    clusters = [[] for i in range(np.unique(clustering.labels_).shape[0]-1)]
    for idx in range(len(kps)):
        if clustering.labels_[idx]!=-1:
            clusters[clustering.labels_[idx]].append((int(kps[idx].pt[0]),int(kps[idx].pt[1])))
    for points in clusters:
        if len(points)>1:
            for idx1 in range(len(points)):
                for idx2 in range(idx1+1,len(points)):
                    cv2.line(forgery,points[idx2],points[idx1],(255,0,0),5)
    plot_image(forgery)
#Firs let us extract SIFT features
key_points,descriptors=siftDetector(tampered1)
show_sift_features(tampered1,key_points)
#Now Let's make clusters and locate forgery

clusters=make_clusters(descriptors)
locate_forgery(tampered1,clusters,key_points)
tampered=cv2.imread(tampered_images[20])
key_points,descriptors=siftDetector(tampered)
clusters=make_clusters(descriptors)
locate_forgery(tampered,clusters,key_points)

# Change Eps parameter to mark more/less features
clusters=make_clusters(descriptors,eps=80)
locate_forgery(tampered,clusters,key_points)
tampered=cv2.imread(tampered_images[50])
key_points,descriptors=siftDetector(tampered)
clusters=make_clusters(descriptors)
locate_forgery(tampered,clusters,key_points)

# Change Eps parameter to mark more/less features
clusters=make_clusters(descriptors,eps=80)
locate_forgery(tampered,clusters,key_points)