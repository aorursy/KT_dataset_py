import cv2

import numpy as np

from matplotlib import pyplot as plt

import pandas as pd

from PIL import Image

from tqdm.notebook import tqdm

import time

import glob

import copy

import torch

!pip install facenet-pytorch

#!git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch

from facenet_pytorch import MTCNN

import pickle

import itertools

import os

import gc



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

device = 'cpu'

dir='/kaggle/input'

dir2='/kaggle/working'
def detect_face(sample,interval=20,normalise=False):

    reader = cv2.VideoCapture(sample)

    images=[]

    tmp0,tmp1=1000,-1000

    for i in range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT))):

        _, image = reader.read()

        try:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except:

            continue

        images.append(image)

        if normalise:

            tmp0=min(tmp0,np.min(image))

            tmp1=max(tmp1,np.max(image))

    rd_frames = np.linspace(0, len(images)-1, interval).astype("int32")

    if normalise:

        tmp2=1/(tmp1-tmp0)

        tmp=[(255.99*(images[e]-tmp0)*tmp2).astype("uint8") for e in rd_frames]

    else:

        tmp=[images[e] for e in rd_frames]        

    imgs_pil = [Image.fromarray(e) for e in tmp]

    box,p=[],[]

    for i in range(len(imgs_pil)):

        try:

            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            detector = MTCNN(device=device, keep_all=True, post_process=False)

            box2,p2=detector.detect(imgs_pil[i])

        except:

            device = 'cpu'

            detector = MTCNN(device=device, keep_all=True, post_process=False)

            box2,p2=detector.detect(imgs_pil[i])

        box.append(box2)

        p.append(p2)       

    reader.release()

    return images,box,p,rd_frames
def divide_face(box,p,t,w0=10,pt=0.99,nframe0=0.5):

    p_ave, ns = 0., 0

    for i in range(len(p)):

        if len(p[i])<1:

            continue

        if p[i][0] is not None:

                p_ave+=max(p[i])

                ns+=1

    p_ave=p_ave/len(p)

    

    p0, nface = p_ave*pt, 0

    for i in range(len(p)):

        if len(p[i])<1:

            continue

        tmp=[]

        if p[i][0] is not None:

            dx, dy = box[i][:,2]-box[i][:,0], box[i][:,3]-box[i][:,1]

            for j in range(len(p[i])):

                if (p[i][j]>=p0) & (dx[j]>=w0) & (dy[j]>=w0):

                    tmp.append(j)

        if p[i][0] is None or len(tmp)<1:

            box[i], p[i] = [], []

        else:

            box[i], p[i] = box[i][tmp], p[i][tmp]

        if len(p[i])>nface:

            nface=len(p[i])



#    xp, yp = (np.arange(nface)-nface)*1000, np.zeros(nface)

    xp, yp = np.zeros(nface,dtype="float32"), np.zeros(nface,dtype="float32")

    face_on=np.zeros(nface,dtype="int32")

    f_par = [[] for i in range(nface)]

    for i in range(len(p)):

        if len(p[i])<1:

            continue

        xc, yc = (box[i][:,0]+box[i][:,2])/2, (box[i][:,1]+box[i][:,3])/2

        #perms = itertools.permutations(range(nface),len(p[i]))

        nface2=max(np.sum(face_on),len(p[i]))

        perms = itertools.permutations(range(nface2),len(p[i]))

        d0 = 10.**10

        for perm in perms:

            tmp=list(perm)

            d = sum(((xc-xp[tmp])**2+(yc-yp[tmp])**2)*face_on[tmp])

            if d<d0:

                d0, perms0 = d, tmp

        if d0 > 700**2*np.sum(face_on):

            continue

        xp[perms0],yp[perms0],face_on[perms0]=xc,yc,1

        for j, k in enumerate(perms0):

            f_par[k].append([box[i][j,0],box[i][j,1],box[i][j,2],box[i][j,3],t[i]])

    f_par = [e for e in f_par if len(e)>=max(len(p)*nframe0,1)]

    for k in range(len(f_par)):

        f_par[k] = np.array(f_par[k],dtype=np.float32).T

    #print(p)

    return f_par
def mk_input(images,f_par,int_point=1):

    t=np.arange(len(images),dtype="int32")

    xsize,ysize=images[0].shape[1],images[0].shape[0]

    nface=len(f_par)

    face=[]

    for i in range(nface):

        ts = (f_par[i][-1,:]).astype("int32")

        p0,p2=np.zeros(len(t),dtype="int32"),np.zeros(len(t),dtype="int32")

        p1,p3=np.zeros(len(t),dtype="int32"),np.zeros(len(t),dtype="int32")

        for n in range(len(ts)+1):

            if n == 0:

                if ts[n]==0:

                    continue

                tl=t[:ts[n]]

            elif n == len(ts):

                tl=t[ts[n-1]:]

            else:

                tl=t[ts[n-1]:ts[n]]

            if len(tl)<=20 and n>0 and n<len(ts):

                n0,n1=max(n-int_point,0),min(n+int_point,len(ts))

            else:

                n0,n1=max(n-1,0),min(n+1,len(ts))

            coe=np.polyfit(ts[n0:n1],f_par[i][:,n0:n1].T,n1-n0-1).T                

            q0,q2=np.poly1d(coe[0])(tl).T, np.poly1d(coe[2])(tl).T

            q1,q3=np.poly1d(coe[1])(tl).T, np.poly1d(coe[3])(tl).T

            p0[tl],p2[tl]=(q0+0.5).astype("int32"),(q2+0.5).astype("int32")

            p1[tl],p3[tl]=(q1+0.5).astype("int32"),(q3+0.5).astype("int32")



        p0,p1=np.where(p0<0,0,p0),np.where(p1<0,0,p1)

        p2,p3=np.where(p2>xsize-1,xsize-1,p2),np.where(p3>ysize-1,ysize-1,p3)



        tmp,tmp0,tmp1=[],np.ones(3)*1000,-np.ones(3)*1000

        for n in range(len(t)):

            tmp.append(images[n][p1[n]:p3[n],p0[n]:p2[n]])

            for j in range(3):

                tmp0[j]=min(tmp0[j],np.min(tmp[n][:,:,j]))

                tmp1[j]=max(tmp1[j],np.max(tmp[n][:,:,j]))

        for n in range(len(t)):

            tmp2=tmp[n].astype("float32")

            for j in range(3):

                tmp2[:,:,j]=((tmp2[:,:,j]-tmp0[j])/(tmp1[j]-tmp0[j]))

            tmp[n]=cv2.resize(tmp2,(150, 200))

        face.append((255.99*np.stack(tmp)).astype("uint8"))

    return face
def mk_movie(face,filename):

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

    video = cv2.VideoWriter(filename, fourcc, 30.0, (face.shape[2],face.shape[1]))

    for i in range(len(face)):

        img=face[i][:,:,[2,1,0]]

        video.write(img)

    video.release()
def mk_dataset(sample):

    t0 = time.time()

    images,box,p,rd_frames=detect_face(sample,interval=20,normalise=False)

    t1 = time.time()

    f_par = divide_face(box,p,rd_frames,pt=0.99)

    gc.collect()

    if len(f_par)<1:

        print('no face')

        t0 = time.time()

        del images

        images,box,p,rd_frames=detect_face(sample,interval=60,normalise=True)

        t1 = time.time()

        f_par = divide_face(box,p,rd_frames,pt=0.9,nframe0=0.)

        #f_par = divide_face(box,p,rd_frames,pt=0.2,nframe0=0.)

    elif len(f_par[0][0])<10:

        print('few faces',len(f_par[0][0]))

        t0 = time.time()

        del images

        images,box,p,rd_frames=detect_face(sample,interval=60,normalise=True)

        t1 = time.time()

        f_par = divide_face(box,p,rd_frames,pt=0.9,nframe0=0.2)

    face = mk_input(images,f_par)

    del images

    t2 = time.time()

    print(sample[-14:-4],f'{t1-t0:.3f} seconds',f'{t2-t1:.3f} seconds')

    return face
os.makedirs(dir2+'/train_sample',exist_ok=True)

files = sorted(glob.glob(dir+'/train_sample/*.pickle'))

df = pd.read_json(dir+'/deepfake-detection-challenge/train_sample_videos/metadata.json').T

samples = (dir+'/deepfake-detection-challenge/train_sample_videos/'+df.index.values).tolist()

video1 = cv2.VideoWriter(dir2+"/train_sample/fake.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), 30.0, (150,200))

video0 = cv2.VideoWriter(dir2+"/train_sample/real.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), 30.0, (150,200))

for n in tqdm(range(len(df))):

    filename=dir2+'/train_sample/'+samples[n][-14:-4]+'-'+df["label"][n]

    if df["label"][n]=='FAKE':

        filename+='-'+df["original"][n][0:10]

    face=mk_dataset(samples[n])

    torch.cuda.empty_cache()

    gc.collect()

    #with open(filename+'.pickle', 'wb') as f:

    #    pickle.dump(face , f)

    for i in range(len(face)):

        t_idx=(np.linspace(0,len(face[i])-1,30)+0.5).astype('int32')

        imgs=face[i][t_idx]

        for j in range(len(imgs)):

            img = cv2.cvtColor(imgs[j], cv2.COLOR_BGR2RGB)

            if df["label"][n]=='FAKE':

                video1.write(img)

            else:

                video0.write(img)

video1.release()

video0.release()
from IPython.display import Video

Video(dir2+"/train_sample/fake.mp4", "mp4")
Video(dir2+"/train_sample/real.mp4", "mp4")