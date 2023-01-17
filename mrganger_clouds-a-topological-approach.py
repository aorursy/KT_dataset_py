%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

matplotlib.rcParams['figure.figsize'] = [15,15]
def read_train():

    df = pd.read_csv('../input/understanding_cloud_organization/train.csv')

    df = df.dropna()

    df[['fname','label']] = df.Image_Label.str.split('_',expand=True)

    df.EncodedPixels = df.EncodedPixels.str.split().apply(np.array,dtype=int)

    df = df.set_index(['fname','label']).EncodedPixels

    df = df.loc[~df.index.duplicated(keep='first')]

    df = df.unstack('label')

    return df
df = read_train()

df.head()
def load_image(fname,test=False):

    from PIL import Image

    path = '../input/understanding_cloud_organization/{}_images/{}'.format('train' if not test else 'test', fname)

    return np.asarray(Image.open(path))
fname = df.index[0]

img = load_image(fname)

plt.imshow(img);
def transform_image(x,thresh,rescale,blur):

    import numpy as np

    from scipy.ndimage import zoom,gaussian_filter,distance_transform_edt

    # normalize between 0 and 1

    img = np.asarray(x,dtype=float)/255



    # take average of RGB channels

    img = img.mean(axis=2)



    # Shrink image

    img = zoom(img,rescale)



    # Remove artifacts from shrinking

    img = gaussian_filter(img,blur)



    # threshold image

    img = (img < thresh).astype(int)



    # Compute distance from black pixels to nearest white pixels (and normalize distances)

    img = distance_transform_edt(img) - (distance_transform_edt(1-img)-1).clip(0,None)

    img = img/rescale



    return img
timg = transform_image(img,thresh=0.5,rescale=0.2,blur=0.1)

plt.imshow(timg,cmap='coolwarm')

plt.clim([-50,50])
plt.imshow(np.block([[timg < t for t in ts] for ts in np.linspace(-20,20,16).reshape(4,4)]), cmap='gray');
from ripser import lower_star_img as lower_star



def pers_image(dgm,res,rng,spread):

    from scipy.ndimage import gaussian_filter,zoom

    

    # birth and death

    idx = np.where(~np.isinf(dgm).any(axis=1))

    b,d = dgm[idx].T

    

    # compute histogram at 3x resolution

    img = np.histogram2d(b,d,bins=res*3,range=[[-rng,rng],[-rng,rng]])[0]

    

    # apply blurring to histogram (relative to total shape)

    img = gaussian_filter(img, np.array(img.shape)*spread)

    

    # decimate histogram (hopefully the blurring prevents aliasing)

    img = img[::3,::3]

    

    return img
dgm = lower_star(timg)

pim = pers_image(dgm,res=500,rng=20,spread=0.02)

plt.imshow(pim.T,origin='lower',cmap='jet');
def apply_mask(img,mask):

    px1 = mask[0::2]

    px2 = px1 + mask[1::2]

    mask = np.zeros(img.shape[:2][::-1],dtype=int)

    mask.flat[px1] = 1

    mask.flat[px2[px2<len(mask.flat)]] = -1

    mask.flat = np.cumsum(mask.flat)

    mask = mask.T

    return img*(mask>0)[:,:,None]



def training_vectors(df,thresh,rescale,blur,res,spread,rng):

    from tqdm import tqdm_notebook

    todo = [(label,col,fname,mask) for label,col in df.items() for fname,mask in col.dropna().items()]

    vecs,labels = [],[]

    for label,col,fname,mask in tqdm_notebook(todo):

        img = load_image(fname)

        img = apply_mask(img,mask)

        img = transform_image(img,thresh,rescale,blur)

        dgm = lower_star(img)

        img = pers_image(dgm,res,rng,spread)

        vecs += [img]

        labels += [label]

    return np.array(vecs),np.array(labels)
fname2 = df.index[3]

img2 = apply_mask(load_image(fname2),df.Flower.iloc[3])

plt.imshow(img2); plt.show();
X,y = training_vectors(df.sample(1000),thresh=0.5,rescale=0.2,blur=0.1,res=20,rng=20,spread=0.02)
from sklearn.linear_model import LogisticRegression

from scipy.ndimage import gaussian_filter

lg = LogisticRegression(C=1e-1,solver='lbfgs',multi_class='multinomial',max_iter=300)

Xi = gaussian_filter(X,(0,1,1)).reshape(-1,20**2)

idx = np.arange(len(Xi))

np.random.shuffle(idx)

i1,i2 = idx[:-300],idx[-300:]

lg.fit(Xi[i1],y[i1])

lg.score(Xi[i2],y[i2])
from scipy.ndimage import zoom

fig,axes = plt.subplots(2,2,figsize=[15,15])

for ci,ax in zip(lg.coef_,axes.flat):

    ax.imshow(zoom(ci.reshape(20,20).T,30),origin='lower',cmap='RdBu');
from sklearn.svm import LinearSVC



svm = LinearSVC(max_iter=500,C=1e-1)

svm.fit(Xi[i1],y[i1])

svm.score(Xi[i2],y[i2])
fig,axes = plt.subplots(2,2,figsize=[15,15])

for ci,ax in zip(svm.coef_,axes.flat):

    ax.imshow(zoom(ci.reshape(20,20).T,30),origin='lower',cmap='RdBu');
def segment(img,size,stride,thresh,rescale,blur,res,rng,spread):

    from tqdm import tqdm_notebook

    from itertools import product

    trans = transform_image(img,thresh,rescale,blur)

    preds = np.empty(np.array(img.shape[:2])//stride+1,dtype='U10')

    for i,j in tqdm_notebook(list(product(range(0,img.shape[0],stride),range(0,img.shape[1],stride)))):

        i1,i2,j1,j2 = ((np.array([[i],[j]])+[-size/2,size/2])*rescale).astype(int).flat

        small_img = trans[i1:i2,j1:j2]

        dgm = lower_star(small_img)

        pim = pers_image(dgm,res,rng,spread)

        preds[i//stride,j//stride] = svm.predict(pim.reshape(1,-1))[0]

    return preds
si = 121

simg = load_image(df.index[si])

seg = segment(simg,size=500,stride=50,thresh=0.5,rescale=0.2,blur=0.1,res=20,rng=20,spread=0.02)



fig,axes = plt.subplots(4,2,figsize=[15,20])

dice = 0

for ax,label in zip(axes,'Flower Fish Gravel Sugar'.split()):

    segl = seg == label

    try:

        ax[0].imshow(apply_mask(img,df[label].iloc[si]));

        segY = apply_mask(np.ones_like(img,dtype=int),df[label].iloc[si])[:,:,0] == 1

    except TypeError:

        segY = np.zeros(img.shape[:2],dtype=bool)

        ax[0].imshow(np.zeros_like(img))

    segX = (zoom(segl.astype(float),50)[:img.shape[0],:img.shape[1]] > 0.5) == 1

    if (segX|segY).sum() == 0:

        dice += 0.25

    else:

        dice += (2*(segX&segY).sum()/(segX|segY).sum())/4

    ax[1].imshow(segX,cmap='Blues');

    ax[0].set_title(label)

print(dice)