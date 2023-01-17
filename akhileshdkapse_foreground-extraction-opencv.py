# Don't worry about this cell variables, they are just forecasting future results..
plt.figure(figsize=(24,12))
plt.imshow(np.hstack([im1/255.0, im2, im4/255.0, im5/255.0]))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
import os
df = pd.read_csv('/kaggle/input/humpback-whale-identification/train.csv')
df.head()
def load(path, size=128):
    img= cv2.resize(cv2.imread(path),(size,size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show():
    f, ax = plt.subplots(3, 5, figsize=(40,20))
    for i in tqdm(range(15)):
        path= os.path.join('../input/humpback-whale-identification/train', df.Image[i])
        img_id= df.Id[i]
        ax[i//5][i%5].imshow(load(path, 300), aspect='auto')
        ax[i//5][i%5].set_title(img_id)
        ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()
    
show()

def adaptive_hist(img, clipLimit= 4.0):
    window= cv2.createCLAHE(clipLimit= clipLimit, tileGridSize=(8, 8))
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    ch1, ch2, ch3 = cv2.split(img_lab)
    img_l = window.apply(ch1)
    img_clahe = cv2.merge((img_l, ch2, ch3))
    return cv2.cvtColor(img_clahe, cv2.COLOR_Lab2BGR)


def show_adhist(clipLimit=4.0):
    f, ax = plt.subplots(3, 5, figsize=(40,20))
    for i in tqdm(range(15)):
        path= os.path.join('../input/humpback-whale-identification/train', df.Image[i])
        img_id= df.Id[i]
        img=load(path, 300)
        img= adaptive_hist(img, clipLimit)
        ax[i//5][i%5].imshow(img, aspect='auto')
        ax[i//5][i%5].set_title(img_id)
        ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()
show_adhist(2.0)

from sklearn.cluster import KMeans
def k_means(img, n_colors= 4):
    w, h, d = original_shape = tuple(img.shape)
    img= img/255.0
    image_array = np.reshape(img, (w * h, d))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)
    
    """Recreate the (compressed) image from the code book & labels"""
    codebook= kmeans.cluster_centers_
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image
def show_kmean(n_colors=4):
    f, ax = plt.subplots(3, 5, figsize=(40,20))
    for i in tqdm(range(15)):
        path= os.path.join('../input/humpback-whale-identification/train', df.Image[i])
        img_id= df.Id[i]
        img=load(path, 300)
        img= k_means(img , n_colors= n_colors)
        ax[i//5][i%5].imshow(img, aspect='auto')
        ax[i//5][i%5].set_title(img_id)
        ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()
show_kmean(n_colors= 4)

def show_edges(n_colors=4):
    f, ax = plt.subplots(3, 5, figsize=(40,20))
    for i in tqdm(range(15)):
        path= os.path.join('../input/humpback-whale-identification/train', df.Image[i])
        img_id= df.Id[i]
        img=load(path, 300)
        img= k_means(img , n_colors= n_colors)
        
        img_gray= cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY)
        img_gray= cv2.medianBlur(img_gray,5)
        edges = cv2.Canny(img_gray,100,200)
        ax[i//5][i%5].imshow(edges, aspect='auto')
        ax[i//5][i%5].set_title(img_id)
        ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()
show_edges(n_colors =3)

def find_box(edges):
    #contour masking
    co, hi = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    con=max(co,key=cv2.contourArea)
    conv_hull=cv2.convexHull(con)
    
    top=tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom=tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left=tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right=tuple(conv_hull[conv_hull[:,:,0].argmax()][0])
    
    return top, bottom, left, right

def draw_bound_box():
    f, ax = plt.subplots(3, 5, figsize=(40,20))
    for i in tqdm(range(15)):
        path= os.path.join('../input/humpback-whale-identification/train', df.Image[i])
        img_id= df.Id[i]
        img=load(path, 300)
        org=img.copy()
        img= k_means(img , n_colors= 10)
        
        img_gray= cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY)
        img_gray= cv2.medianBlur(img_gray,7)
        edges = cv2.Canny(img_gray,100,200)
        
        kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        top,bottom,left,right = find_box(edges)
        org=cv2.rectangle(org, (left[0], top[1]), (right[0], bottom[1]), (0, 255, 0), thickness=3)
        
        ax[i//5][i%5].imshow(org, aspect='auto')
        ax[i//5][i%5].set_title(img_id)
        ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()
    
    
draw_bound_box()

def forgrd_ext(img, rec):
    mask= np.zeros(img.shape[:2], np.uint8)
    bgmodel= np.zeros((1, 65), np.float64)
    fgmodel= np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rec, bgmodel, fgmodel, 3, cv2.GC_INIT_WITH_RECT)
    mask2= np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img= img*mask2[:,:,np.newaxis]
    img[np.where((img == [0,0,0]).all(axis = 2))] = [255.0, 255.0, 255.0]
    return img

def ext_frgd():
    f, ax = plt.subplots(5, 5, figsize=(40,30))
    for i in tqdm(range(25)):
        path= os.path.join('../input/humpback-whale-identification/train', df.Image[i])
        img_id= df.Id[i]
        img=load(path, 300)
        org=img.copy()
        img= k_means(img , n_colors= 10)
        
        img_gray= cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY)
        img_gray= cv2.medianBlur(img_gray,7)
        edges = cv2.Canny(img_gray,100,200)
        
        kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        top,bottom,left,right = find_box(edges)
        rec= (left[0], top[1], right[0]-left[0], bottom[1]-top[1])
        forground_img= forgrd_ext(org, rec)
        
        ax[i//5][i%5].imshow(forground_img, aspect='auto')
        ax[i//5][i%5].set_title(img_id)
        ax[i//5][i%5].set_xticks([]); ax[i//5][i%5].set_yticks([])
    plt.show()
    

ext_frgd()
path= os.path.join('../input/humpback-whale-identification/train', df.Image[7])
im1= load(path, 300)

im2= img= k_means(im1 , n_colors= 3)

im3= cv2.cvtColor(np.uint8(im2*255), cv2.COLOR_RGB2GRAY)
im3= cv2.medianBlur(im3,5)
im3 = cv2.Canny(im3,100,200)
kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
im31 = cv2.morphologyEx(im3, cv2.MORPH_CLOSE, kernel)

top,bottom,left,right = find_box(im31)
im4=cv2.rectangle(im1.copy(), (left[0], top[1]), (right[0], bottom[1]), (0, 255, 0), thickness=3)

rec= (left[0], top[1], right[0]-left[0], bottom[1]-top[1])
im5= forgrd_ext(im1, rec)
