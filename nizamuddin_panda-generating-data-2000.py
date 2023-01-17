import os

import cv2

import skimage.io

from tqdm.notebook import tqdm

import zipfile

import numpy as np
TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'

MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'

OUT_TRAIN = 'train.zip'

LABELS    = '../input/prostate-cancer-grade-assessment/train.csv'
!rm -rf ./train.zip
# def get_tiles(img, n_tiles,tile_size,mode=0):

#     '''

#     from 36, 256x256

#     '''

#     result = []

#     h, w, c = img.shape

#     pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)

#     pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

#     img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values=255)

#     img3 = img2.reshape(

#         img2.shape[0] // tile_size,

#         tile_size,

#         img2.shape[1] // tile_size,

#         tile_size,

#         3

#     )

#     img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)

#     n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()

#     if len(img3) < n_tiles:

#         img3 = np.pad(img3,[[0,n_tiles-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)

#     idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]

#     img3 = img3[idxs]

#     for i in range(len(img3)):

#         result.append({'img':img3[i], 'idx':i})

#     return result, n_tiles_with_info >= n_tiles
# def tile(img,N,sz):

#     result = []

#     shape = img.shape

#     #paddings

#     pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz

#     #images

#     img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=255)

#     img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)

#     img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

#     if len(img) < N:

#         img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

#     idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]

#     img = img[idxs]

#     for i in range(len(img)):

#         result.append({'img':img[i],'idx':i})

#     return result
# def display_image(img,is_mask=False):

#     '''

#     To display image/mask 

#     args: img, image

#           is_mask, boolean True if greyscale mask is passed

#     '''

#     from matplotlib import pyplot as plt

#     %matplotlib inline

#     if is_mask:

#         plt.imshow(img,cmap='gray')

#     else:

#         plt.imshow(img)

#     plt.show()     
# img  = skimage.io.MultiImage(os.path.join(TRAIN,'cdd5b7d07b98f61d3668207531b4de07'+'.tiff'))[-1]

# tiles=tile_high_res(os.path.join(TRAIN,'085a35715e8a0f0edeccff03290a6baf'+'.tiff'))

# for t in tiles:

#     img,idx = t['img'],t['idx']

#     print(t['img'].shape,t['idx'])

#     display_image(t['img'])
# def tile(img, mask):

#     result = []

#     shape = img.shape

#     #paddings

#     pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz

#     #images

#     img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=255)

#     img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)

#     img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

#     #masks

#     mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=0)

#     mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)

#     mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

#     if len(img) < N:

#         mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)

#         img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

#     idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]

#     img = img[idxs]

#     mask = mask[idxs]

#     for i in range(len(img)):

#         result.append({'img':img[i], 'mask':mask[i], 'idx':i})

#     return result

# x_tot,x2_tot = [],[]

# names = [name[:-10] for name in os.listdir(MASKS)]

# with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out,zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:

#     for name in tqdm(names):

#         img  = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[-1]

#         mask = skimage.io.MultiImage(os.path.join(MASKS,name+'_mask.tiff'))[-1]

#         tiles = tile(img,mask)

#         for t in tiles:

#             img,mask,idx = t['img'],t['mask'],t['idx']

#             x_tot.append((img/255.0).reshape(-1,3).mean(0))

#             x2_tot.append(((img/255.0)**2).reshape(-1,3).mean(0)) 

#             #if read with PIL RGB turns into BGR

#             img = cv2.imencode('.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]

#             img_out.writestr(f'{name}_{idx}.png', img)

#             mask = cv2.imencode('.png',mask[:,:,0])[1] 

#             mask_out.writestr(f'{name}_{idx}.png', mask)
def tile_high_res(fname):

    import gc

    N = 16

    sz= 256

    import openslide

    # use layer 2 for tile selection

    img = skimage.io.MultiImage(fname)[-1]

    shape = img.shape

    r = 16 # ratio of layer 0 vs layer 2 res

    sz16 = sz//r

    pad0,pad1 = (sz16 - shape[0]%sz16)%sz16, (sz16 - shape[1]%sz16)%sz16

    img  = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=255)

    img  = img.reshape(img.shape[0]//sz16,sz16,img.shape[1]//sz16,sz16,3)

    img  = img.transpose(0,2,1,3,4).reshape(-1,sz16,sz16,3)

    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:min(N,len(img))]

    del img

    gc.collect()

    # read layer 0 tile by tile with use of openslide

    n0,n1 = (pad0+shape[0])//sz16, (pad1+shape[1])//sz16

    img0  = openslide.OpenSlide(fname)

    tiles = []

    for idx in idxs:

        x = (-pad0//2 + sz16*(idx//n1))*r

        y = (-pad1//2 + sz16*(idx%n1))*r

        t = np.array(img0.read_region((y,x),0,(sz,sz)))[:,:,:3]

        tiles.append(t)

    del img0

    gc.collect()

    for i in range(N - len(tiles)): 

        tiles.append(np.full((sz,sz,3), 255, dtype=np.uint8))

    result = []

    for i in range(len(tiles)):

        result.append({'img':tiles[i],'idx':i})

    del tiles

    gc.collect()

    return result

#     return np.stack(tiles)
# from joblib import Parallel,delayed

# x_tot,x2_tot = [],[]

# imgs=[]

# full_names = []

# names = set([name[:-10] for name in os.listdir(MASKS)])-set(to_drop)

# for name in tqdm(list(names)):

#     full_names.append(os.path.join(TRAIN,name+'.tiff'))

# #res=Parallel(n_jobs=8,backend='threading')(delayed(tile_high_res)(name) for name in tqdm(full_names))
# with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:

#     for name in tqdm(full_names):

#         tile_high_res(name,img_out)



# from kaggle_datasets import KaggleDatasets

# GCS_DS_PATH = KaggleDatasets().get_gcs_path('prostate-cancer-grade-assessment')
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>IDEMPOTENT

import pandas as pd

susp = pd.read_csv('../input/suspicious-panda/PANDA_Suspicious_Slides.csv')

# ['marks', 'No Mask', 'Background only', 'No cancerous tissue but ISUP Grade > 0', 'tiss', 'blank']

to_drop = susp.query("reason in ['marks','Background only','tiss','blank']")['image_id']

print("len(todrop):",len(to_drop))

df = pd.read_csv(LABELS).set_index('image_id')

good_index = list(set(df.index)-set(to_drop))

df = df.loc[good_index]

df = df.reset_index()

df = pd.concat([df.query('isup_grade==0').iloc[:1200],df.query('isup_grade==1').iloc[:1200],df.query('isup_grade==2 or isup_grade==3 or isup_grade==4 or isup_grade==5')],axis=0)

df = df.sample(n=2000,random_state=2020).reset_index(drop=True)#shuffling

df[['isup_grade']].hist(bins=50)

names = df['image_id']

full_names = []

for name in tqdm(names):

    full_names.append(os.path.join(TRAIN,name+'.tiff'))

print("len(full_names):",len(full_names)," these are used to genrate tiles further...")
x_tot,x2_tot = [],[]

with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:

    for full_path in tqdm(full_names):

        tiles = tile_high_res(full_path)

        name = full_path.split("/")[-1].split(".")[0]

        for t in tiles:

            img,idx = t['img'],t['idx']

            x_tot.append((img/255.0).reshape(-1,3).mean(0))

            x2_tot.append(((img/255.0)**2).reshape(-1,3).mean(0)) 

            #-if read with PIL RGB turns into BGR

            img = cv2.imencode('.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]

            img_out.writestr(f'{name}_{idx}.png', img)

        del tiles

        import gc

        gc.collect()
#image statss........

img_avr =  np.array(x_tot).mean(0)

img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)

print('mean:',img_avr, ', std:', np.sqrt(img_std),"x2_tot:",np.array(x2_tot).mean(0))     
del names

import gc

gc.collect()