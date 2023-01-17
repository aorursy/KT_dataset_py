import numpy as np
import matplotlib.pyplot as plt
import cv2
import os



f = open("../input/depths.csv")
data=f.readlines()
f.close()

fils=[]
dep=[]

for i in data[1:]:
    fils.append(i.split(',')[0])
    dep.append(float(i.split(',')[1]))
    
    
dir_='../input/'

fnames=[]
imag=[]
r=[]
l=[]
u=[]
d=[]
dept=[]
mask=[]

#No mask generation
ema=np.zeros((101,101))
ema[0,:]=255

for filename in os.listdir(dir_+'train/images'):
    fnames.append(filename)
    fimg=cv2.imread(dir_+'train/images/'+filename,0)
    mask.append(cv2.imread(dir_+'/train/masks/'+filename,0))
    imag.append(fimg)
    r.append(fimg[:,-1])
    l.append(fimg[:,0])
    dept.append(dep[fils.index(filename.split('.')[0])])

for filename in os.listdir(dir_+'test/images'):
    fnames.append(filename)
    fimg=cv2.imread(dir_+'test/images/'+filename,0)
    mask.append(ema)
    imag.append(fimg)
    r.append(fimg[:,-1])
    l.append(fimg[:,0])
    dept.append(dep[fils.index(filename.split('.')[0])])
    

m=np.random.randint(22000,size=1)
m=[20449,4439, 18152, 6156, 16461,3535,8944,15799,21609,453]
for x in m:

    corr_r=[]
    corr_l=[]

    for i in range(0,len(l)):
        if i==x:
            corr_r.append(0)
        else:
            corr_r.append(np.corrcoef(r[x],l[i])[0,1])

    # Finding index of max correlation
    ri=corr_r.index(max(corr_r))

    # Normalization of images
    a=imag[x]-np.mean(imag[x])
    b=imag[ri]-np.mean(imag[ri])

    mina=min([np.amin(a),np.amin(b)])
    maxa=min([np.amax(a),np.amax(b)])

    plt.figure(figsize=(10,7))
    plt.subplot(121)
    plt.imshow(a,cmap='gray',vmin=mina,vmax=maxa)
    plt.imshow(mask[x],cmap='Greens',vmin=0,vmax=255,alpha=0.2)
    plt.axis('off')
    plt.text(50, 100, dept[x], fontsize=12)
    plt.subplot(122)
    plt.imshow(b,cmap='gray',vmin=mina,vmax=maxa)
    plt.imshow(mask[ri],cmap='Greens',vmin=0,vmax=255,alpha=0.2)
    plt.axis('off')
    plt.text(0, 10, str(max(corr_r))[:5], fontsize=12)
    plt.text(50, 100, dept[ri], fontsize=12)

    # Tight layout of images
    plt.suptitle(str(x))
    plt.subplots_adjust(wspace=0, hspace=0)
m=[3535]
for x in m:

    corr_r=[]
    corr_l=[]

    for i in range(0,len(l)):
        if i==x:
            corr_r.append(0)
        else:
            corr_r.append(np.corrcoef(r[x],l[i])[0,1])

    # Finding index of max correlation
    ri=corr_r.index(max(corr_r))

    # Normalization of images
    a=imag[x]-np.mean(imag[x])
    b=imag[ri]-np.mean(imag[ri])

    mina=min([np.amin(a),np.amin(b)])
    maxa=min([np.amax(a),np.amax(b)])

    plt.figure(figsize=(10,7))
    plt.subplot(121)
    plt.imshow(a,cmap='gray',vmin=mina,vmax=maxa)
    plt.imshow(mask[x],cmap='Greens',vmin=0,vmax=255,alpha=0.2)
    plt.axis('off')
    plt.text(50, 100, dept[x], fontsize=12)
    plt.subplot(122)
    plt.imshow(b,cmap='gray',vmin=mina,vmax=maxa)
    plt.imshow(mask[ri],cmap='Greens',vmin=0,vmax=255,alpha=0.2)
    plt.axis('off')
    plt.text(0, 10, str(max(corr_r))[:5], fontsize=12)
    plt.text(50, 100, dept[ri], fontsize=12)

    # Tight layout of images
    plt.suptitle(str(x))
    plt.subplots_adjust(wspace=0, hspace=0)