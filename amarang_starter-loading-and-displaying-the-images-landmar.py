import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from tqdm import tqdm_notebook

import skimage
!ls -lrth ../input/face_images/face_images | head
dfc = pd.read_json("../input/metadata.json/metadata.json")

dfc["landmarks"] = dfc["landmarks"].map(np.array)

dfc.head()
imgs = []

for index in tqdm_notebook(dfc.index):

    img = skimage.io.imread(f"../input/face_images/face_images/{index}.png")

    imgs.append(img)

dfc["image_data"] = imgs
plt.imshow(dfc.image_data.iloc[0])
img = dfc.image_data.iloc[42]

landmarks = dfc.landmarks.iloc[42]

fig,ax=plt.subplots(1, 1, figsize=(4,6),gridspec_kw = {'wspace':0, 'hspace':0, "top":1,"left":0,"right":1,"bottom":0})

ax.imshow(img)

ax.scatter(landmarks[:,0],landmarks[:,1],color="r",s=25)

ax.set_aspect('equal')

clean(ax)

fig.savefig("face.png")

def stack_plot(imgs,rows=None,cols=None,vlandmarks=None,landmark_color="r"):

    N = imgs.shape[0]

    aspect_ratio = 1.0*imgs[0].shape[0]/imgs[0].shape[1]

    rows = rows or int(N**0.5)

    cols = cols or int(N**0.5)

    figsize = [cols,rows*aspect_ratio]

    maxinches = 30

    if figsize[0] > maxinches: figsize = [maxinches,figsize[1]*maxinches/figsize[0]]

    if figsize[1] > maxinches: figsize = [figsize[0]*maxinches/figsize[1],maxinches]

    fig, axs = plt.subplots(rows,cols,figsize=figsize,squeeze=False)

    iterable = zip(imgs,axs.reshape(-1))

    if N > 100: iterable = tqdm_notebook(iterable)

    def clean(ax):

        ax.set_frame_on(False)

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

    for i,(img,ax) in enumerate(iterable):

        ax.imshow(img)

        if vlandmarks is not None and len(vlandmarks) == N:

            ax.scatter(vlandmarks[i][:,0],vlandmarks[i][:,1],color=landmark_color,s=5)

        clean(ax)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
stack_plot(dfc.image_data.values[:100],vlandmarks=dfc.landmarks.values[:100])