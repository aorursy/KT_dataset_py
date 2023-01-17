import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
import sys
sys.path.insert(0, "/kaggle/input/university-project/revolver")

from revolver.data.pascal import VOCSemSeg, VOCInstSeg, SBDDSemSeg, SBDDInstSeg
from revolver.data.seg import MaskSemSeg, MaskInstSeg
from revolver.data.sparse import SparseSeg
from revolver.data.filter import TargetFilter, TargetMapper
import os
print(os.listdir('../input'))
#print(os.listdir('/kaggle/working/benchmark_RELEASE/dataset'))
#import os, sys, tarfile

#def extract(tar_url, extract_path='/kaggle/working'):
#    print (tar_url)
#    tar = tarfile.open(tar_url, 'r')
#    for item in tar:
#        tar.extract(item, extract_path)
#        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
#            extract(item.name, "./" + item.name[:item.name.rfind('/')])
#try:

 #   extract("/kaggle/input/data-sbdd/benchmark.tgz")
#    print ('Done.')
#except:
 #   name = os.path.basename(sys.argv[0])
 #   print (name[:name.rfind('.')], '<filename>')
# the default dataset root dir works on its own if invoked from the root dir of the project,
# and this call gives the path only as an illustration.
ds = VOCSemSeg(root_dir='/kaggle/input/pascal-voc-2012/VOC2012')  
print("Dataset with {} images in split {} and {} classes:\n{}".format(len(ds), ds.split, ds.num_classes, ds.classes))

# all datasets return a 3-tuple with image, target/label, and auxiliary dictionary
# note: image and target are np arrays unless they are cast by transforms
im, target, aux = ds[0]  

plt.figure()
plt.title("item ID {}".format(ds.slugs[0]))
plt.imshow(im)
plt.axis('off')
# plot the target class-by-class
# in the next step we'll make this simpler and prettier with a palette to color the classes
plt.figure(figsize=(16, 4))
plt.subplot(1, 3, 1)
plt.imshow(target == ds.classes.index('aeroplane'))
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(target == ds.classes.index('person'))
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(target == ds.ignore_index)
plt.axis('off')
def load_and_show(ds, title, count=1, print_aux=True):
    for i in range(count):
        im, target, aux = ds[np.random.randint(0, len(ds))]
        im = Image.fromarray(im, mode='RGB')
        target = Image.fromarray(target, mode='P')
        target.putpalette(ds.palette)  # plot with pretty colors
        fig = plt.figure(figsize=(12, 12))
        plt.subplot(1, 2, 1)
        if print_aux:
            figtitle = title + " " + str(aux)
        plt.title(title)
        plt.imshow(im)
        plt.axis('off')
        plt.tight_layout()
        plt.subplot(1, 2, 2)
        plt.imshow(target)
        plt.axis('off')
ds = VOCSemSeg(root_dir='/kaggle/input/pascal-voc-2012/VOC2012')  # default split is train
load_and_show(ds, "VOC semantic segmentation")

ds = VOCInstSeg(split='val',root_dir='/kaggle/input/pascal-voc-2012/VOC2012')
load_and_show(ds, "VOC instance segmentation")

ds = SBDDInstSeg(root_dir='/kaggle/input/sbdd-modified/sbdd')
load_and_show(ds, "SBDD instance segmentation")

ds = SBDDSemSeg(split='train',root_dir='/kaggle/input/sbdd-modified/sbdd')
load_and_show(ds, "SBDD semantic segmentation")
ds = MaskSemSeg(VOCSemSeg(root_dir='/kaggle/input/pascal-voc-2012/VOC2012'))
load_and_show(ds, "VOC class masks", count=3)

# instance masking requires both a semantic seg. and an instance seg. dataset
# to preserve class information
ds = MaskInstSeg(VOCSemSeg(root_dir='/kaggle/input/pascal-voc-2012/VOC2012'), VOCInstSeg(root_dir='/kaggle/input/pascal-voc-2012/VOC2012'))
load_and_show(ds, "VOC instance masks", count=3)
# reduce masks to 100 points per class, resampled on every load
ds = VOCSemSeg(root_dir='/kaggle/input/pascal-voc-2012/VOC2012')
ds = SparseSeg(ds, count=100)
load_and_show(ds, "sparse VOC semantic segmentation", count=3)

# reduce masks to a fixed 16 points per class
# re-loading does not choose new points
ds = VOCSemSeg(root_dir='/kaggle/input/pascal-voc-2012/VOC2012')
ds = SparseSeg(ds, count=16, static=True)
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("static sparsity")
im, target, aux = ds[0]
plt.imshow(target)
plt.subplot(1, 2, 2)
im, target, aux = ds[0]
plt.imshow(target)
# filter classes to only load images that include the filtered classes
ds = VOCSemSeg(root_dir='/kaggle/input/pascal-voc-2012/VOC2012')
classes_to_filter = (ds.classes.index('aeroplane'), ds.classes.index('train'), ds.classes.index('car'))
ds_filtered = TargetFilter(ds, classes_to_filter)
load_and_show(ds_filtered, "Planes, Trains, and Automobiles", count=3)

# filter classes to only load images that include the filtered classes, and exclude other classes from the target
ds_mapped = TargetMapper(ds_filtered, {k: k in classes_to_filter for k in range(len(ds.classes))})
load_and_show(ds_mapped, "Planes, Trains, and Automobiles ONLY", count=3, print_aux=False)

# collapse all classes to make foreground/background target 
ds_fgbg = TargetMapper(ds, {k: 1 for k in range(1, ds.num_classes + 1)})
load_and_show(ds_fgbg, "FG/BG", count=3, print_aux=False)
ds = MaskInstSeg(VOCSemSeg(root_dir='/kaggle/input/pascal-voc-2012/VOC2012'), VOCInstSeg(root_dir='/kaggle/input/pascal-voc-2012/VOC2012'))
ds = TargetFilter(ds, (ds.classes.index('cat'),))
ds = SparseSeg(ds, count=128)
load_and_show(ds, "sparse cat instance masks", count=12)
import pickle

cache_path = '/kaggle/working/data/cache/ds-cache.pkl'

pickle.dump(ds, open(cache_path, 'wb'))
del ds

ds = pickle.load(open(cache_path, 'rb'))
os.remove(cache_path)

load_and_show(ds, "sparse cat instance masks, cached", count=12)
