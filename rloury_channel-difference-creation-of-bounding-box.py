import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sympy.solvers import solve
from sympy import Symbol
import math
import cv2

marks = pd.read_csv('../input/train_ship_segmentations.csv') # Markers for ships
images = os.listdir('../input/train') # Images for training
os.chdir("../input/train")
def mask_part(pic):
    '''
    Function that encodes mask for single ship from .csv entry into numpy matrix
    '''
    back = np.zeros(768**2)
    starts = pic.split()[0::2]
    lens = pic.split()[1::2]
    for i in range(len(lens)):
        back[(int(starts[i])-1):(int(starts[i])-1+int(lens[i]))] = 1
    return np.reshape(back, (768, 768, 1))

def is_empty(key):
    '''
    Function that checks if there is a ship in image
    '''
    df = marks[marks['ImageId'] == key].iloc[:,1]
    if len(df) == 1 and type(df.iloc[0]) != str and np.isnan(df.iloc[0]):
        return True
    else:
        return False
    
def masks_all(key):
    '''
    Merges together all the ship markers corresponding to a single image
    '''
    df = marks[marks['ImageId'] == key].iloc[:,1]
    masks= np.zeros((768,768,1))
    if is_empty(key):
        return masks
    else:
        for i in range(len(df)):
            masks += mask_part(df.iloc[i])
        return np.transpose(masks, (1,0,2))
def image_contrast(image):
    shape=np.shape(image)
    im=image.flatten()
    vide=0
    for i in range(shape[0]*shape[1]):
        if im[i]>200:
            im[i]=255
            vide=1
        else:
            im[i]=0
    if vide==0:
        print(-1)
        return [-1]
    
    return np.reshape(im,shape)

def corner_box(image):
    if np.shape(image_contrast(image))[0]>2:
        image=image_contrast(image).flatten()
        list_coord=[]
        for i in range(768*768):
            if image[i]==255:
                x=np.array([(i%768),int((i/768))])
                list_coord.append(x)
        image=np.array(list_coord)
        ca = np.cov(image,y = None,rowvar = 0,bias = 1)

        v, vect = np.linalg.eig(ca)
        tvect = np.transpose(vect)

        ar = np.dot(image,np.linalg.inv(tvect))

        mina = np.min(ar,axis=0)
        maxa = np.max(ar,axis=0)
        diff = (maxa - mina)*0.5

        center = mina + diff
        corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]],center+[-diff[0],-diff[1]]])

        corners = np.dot(corners,tvect)
        corner_swne=[0]*4
        l_n=1000
        l_s=0
        l_w=10000
        l_e=0
        for i in corners:
            if i[1]<l_n:
                l_n=i[1]
                corner_swne[2]=i
            if i[1]>=l_s:
                l_s=i[1]
                corner_swne[0]=i
            if i[0]>=l_e:
                l_e=i[0]
                corner_swne[3]=i
            if i[0]<l_w:
                l_w=i[0]
                corner_swne[1]=i

        return corner_swne
    else:
        return -1

def coeff_direct_box(image):
    if np.shape(image_contrast(image))[0]>2:
        coord=corner_box(image)
        s=coord[0]
        w=coord[1]
        n=coord[2]
        e=coord[3]

        v_ne=[e[0]-n[0],e[1]-n[1]]
        v_es=[s[0]-e[0],s[1]-e[1]]
        v_sw=[w[0]-s[0],w[1]-s[1]]
        v_wn=[n[0]-w[0],n[1]-w[1]]

        vect = [v_ne,v_es,v_sw,v_wn]

        if vect[0][0]!=0:
            c_ne=vect[0][1]/vect[0][0]
        else:
            c_ne=0
        if vect[1][0]!=0:
            c_es=vect[1][1]/vect[1][0]
        else:
            c_es=0
        if vect[2][0]!=0:
            c_sw=vect[2][1]/vect[2][0]
        else:
            c_sw=0
        if vect[3][0]!=0:
            c_wn=vect[3][1]/vect[3][0]
        else:
            c_wn=0

        return c_ne,c_es,c_sw,c_wn
    else:
        return -1

def bounding_box_channel(image):
    if np.shape(image_contrast(image))[0]>2:
        coord=corner_box(image)
        coeff = coeff_direct_box(image)
        image=[0]*768*768

        for i in range(768*768):  
                    if  int(i/768)>coord[2][1] and int(i/768)<coord[0][1] and i%768>coord[1][0]and i%768<coord[3][0]:
                         image[i]=255
                    if (i%768-coord[2][0])!=0 and (int(i/768)-int(coord[2][1]))/(i%768-coord[2][0])<coeff[0] and i%768>coord[2][0]:
                            image[i]=0
                    if (i%768-coord[3][0])!=0 and (int(i/768)-int(coord[3][1]))/(i%768-coord[3][0])<coeff[1] and int(i/768)>coord[3][1]:
                            image[i]=0
                    if (i%768-coord[0][0])!=0 and (int(i/768)-int(coord[0][1]))/(i%768-coord[0][0])<coeff[2] and i%768<coord[0][0]:
                            image[i]=0
                    if (i%768-coord[1][0])!=0 and (int(i/768)-int(coord[1][1]))/(i%768-coord[1][0])<coeff[3] and int(i/768)<coord[1][1]:
                            image[i]=0

        return np.reshape(image,[768,768])
    else:
        return np.reshape([0]*768*768,[768,768])

    
def bounding_box(image,mask):
    image_1=plt.imread(image)[:,:,2]-plt.imread(image)[:,:,1]
    image_2=plt.imread(image)[:,:,1]-plt.imread(image)[:,:,0]
    image_3=plt.imread(image)[:,:,2]-plt.imread(image)[:,:,0]
    bb_1=bounding_box_channel(image_1).flatten()
    bb_2=bounding_box_channel(image_2).flatten()
    bb_3=bounding_box_channel(image_3).flatten()
    bb_4=bb_1+bb_2+bb_3
    bon=0
    mask_f=mask.flatten()
    for i in range(768*768):
        if bb_4[i]>600:
            bb_4[i]=255
        else:
            bb_4[i]=0
        if bb_4[i]==mask_f[i]:
            bon+=1
    
    bb_1=np.reshape(bb_1,[768,768])
    bb_2=np.reshape(bb_2,[768,768])
    bb_3=np.reshape(bb_3,[768,768])
    
    bb_4=np.reshape(bb_4,[768,768])
    plt.subplot(1,3,1)
    plt.imshow(bb_4)
    plt.subplot(1,3,2)
    plt.imshow(mask)
    plt.subplot(1,3,3)
    plt.imshow(plt.imread(image))
    plt.show()
    print("Result {} %".format((bon/(768*768))*100))
sel=26 #the image's number
bounding_box(images[sel],masks_all(images[sel])[:,:,0])
