from fastai.vision import *

import skimage.io

import zipfile
def crop_and_tile(fn, tiff_layer=2, empty_thr=250, tile_thr=0):

    im = skimage.io.MultiImage(str(fn))[tiff_layer]

    crop = tile_size*(im.shape[0]//tile_size), tile_size*(im.shape[1]//tile_size)

    im = im[:crop[0], :crop[1]]

    imr = im.reshape(im.shape[0]//tile_size,tile_size,im.shape[1]//tile_size, tile_size, 3)

    imr = imr.transpose(1,3,0,2,4)

    imr = imr.reshape(imr.shape[0], imr.shape[1], imr.shape[2]*imr.shape[3], imr.shape[4])

    imr = imr.transpose(2,0,1,3)

    not_empty = np.array([(im[...,0]<empty_thr).sum() for im in imr])>tile_thr

    return imr[not_empty]



def save_tiles(path:Path, filename:str, tiles):

    for i, t in enumerate(tiles):

        im = PIL.Image.fromarray(t)

        im.save(path/f'{filename}_{i}.png')
tile_size = 256

path = Path('/kaggle/input/prostate-cancer-grade-assessment')

save_path = Path(f'/kaggle/working/data{tile_size}')

save_path.mkdir(exist_ok=True)

train_folder = 'train_images'

masks_folder = 'train_label_masks'

path.ls()
# Load train.csv

train_df = pd.read_csv(path/'train.csv')

train_df.head()
files = (path/train_folder).ls()



def do_one(fn, *args):

    tiles = crop_and_tile(fn)

    save_tiles(save_path, fn.stem, tiles)

    

parallel(do_one, files, max_workers=4)
# Plot some samples

saved_files = np.random.permutation(save_path.ls())

fig, axes = plt.subplots(ncols=16,nrows=16,figsize=(12,12),dpi=120,facecolor='gray')

for ax, fn in zip(axes.flat, saved_files):

    im = PIL.Image.open(fn)

    ax.imshow(np.array(im))

    ax.axis('off')