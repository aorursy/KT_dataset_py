%matplotlib inline
import cv2
import numpy as np
import pydicom as dicom
import json
import os
import shutil
import sys
import random
from matplotlib import image
import matplotlib.pyplot as plt
import re
from skimage.util.montage import montage2d
montage3d = lambda x, **k: montage2d(np.stack([montage2d(y, **k) for y in x],0))
# number of bins to use in histogram for gaussian regression
NUM_BINS = 100
# number of standard deviatons past which we will consider a pixel an outlier
STD_MULTIPLIER = 2
# number of points of our interpolated dataset to consider when searching for
# a threshold value; the function by default is interpolated over 1000 points,
# so 250 will look at the half of the points that is centered around the known
# myocardium pixel
THRESHOLD_AREA = 250
# number of pixels on the line within which to search for a connected component
# in a thresholded image, increase this to look for components further away
COMPONENT_INDEX_TOLERANCE = 20
# number of angles to search when looking for the correct orientation
ANGLE_SLICES = 36
ALL_DATA_DIR =  os.path.join('..', 'input', 'train', 'train')
X_DIM, Y_DIM = 64, 64
X_DIM, Y_DIM = 128, 128
T_DIM = 30
class Dataset(object):
    dataset_count = 0

    def __init__(self, directory, subdir):
        # deal with any intervening directories
        while True:
            subdirs = next(os.walk(directory))[1]
            if len(subdirs) == 1:
                directory = os.path.join(directory, subdirs[0])
            else:
                break

        slices = []
        for s in subdirs:
            m = re.match("sax_(\d+)", s)
            if m is not None:
                slices.append(int(m.group(1)))

        slices_map = {}
        first = True
        times = []
        for s in slices:
            files = next(os.walk(os.path.join(directory, "sax_%d" % s)))[2]
            offset = None

            for f in files:
                m = re.match("IM-(\d{4,})-(\d{4})\.dcm", f)
                if m is not None:
                    if first:
                        times.append(int(m.group(2)))
                    if offset is None:
                        offset = int(m.group(1))

            first = False
            slices_map[s] = offset

        self.directory = directory
        self.time = sorted(times)
        self.slices = sorted(slices)
        self.slices_map = slices_map
        self.name = subdir

    def _filename(self, s, t):
        return os.path.join(self.directory,"sax_%d" % s, "IM-%04d-%04d.dcm" % (self.slices_map[s], t))

    def _read_dicom_image(self, filename):
        d = dicom.read_file(filename)
        img = d.pixel_array
        return np.array(img)

    def _read_all_dicom_images(self):
        f1 = self._filename(self.slices[0], self.time[0])
        d1 = dicom.read_file(f1)
        (x, y) = d1.PixelSpacing
        (x, y) = (float(x), float(y))
        f2 = self._filename(self.slices[1], self.time[0])
        d2 = dicom.read_file(f2)

        # try a couple of things to measure distance between slices
        try:
            dist = np.abs(d2.SliceLocation - d1.SliceLocation)
        except AttributeError:
            try:
                dist = d1.SliceThickness
            except AttributeError:
                dist = 8  # better than nothing...

        self.images = np.array([[self._read_dicom_image(self._filename(d, i))
                                 for i in self.time]
                                for d in self.slices])
        self.dist = dist
        self.area_multiplier = x * y

    def load(self):
        self._read_all_dicom_images()
base_path = os.path.join(ALL_DATA_DIR, '140')
tData = Dataset(base_path,'140')
tData.load()
%matplotlib inline
plt.imshow(tData.images[0,0,:,:], cmap = 'bone')
fig, ax1 = plt.subplots(1,1, figsize = (20,20))
ax1.imshow(montage3d(tData.images), cmap = 'bone')
from scipy.ndimage import zoom
rezoom = lambda in_data: zoom(in_data.images, [1, 
                                               T_DIM/in_data.images.shape[1], 
                                               X_DIM/in_data.images.shape[2], 
                                               Y_DIM/in_data.images.shape[3]])
image_stack = rezoom(tData)
print(image_stack.shape)
fig, ax1 = plt.subplots(1,1, figsize = (20,20))
ax1.imshow(montage3d(image_stack), cmap = 'bone')
from glob import glob
base_path = os.path.join(ALL_DATA_DIR, '*')
all_series = glob(base_path)
from warnings import warn
def read_and_process(in_path):
    try:
        cur_data = Dataset(in_path,
                           os.path.basename(in_path))
        cur_data.load()
        if cur_data.time is not None:
            zoom_time = zoom(cur_data.time, [T_DIM/len(cur_data.time)])
        else:
            zoom_time = range(T_DIM)
        return [in_path, zoom_time, cur_data.area_multiplier, rezoom(cur_data)]
    except Exception as e:
        warn('{}'.format(e), RuntimeWarning)
        return None
%%time
a,d,b,c = read_and_process(all_series[-100])
print(c.shape)
import dask
import dask.diagnostics as diag
from bokeh.io import output_notebook
from bokeh.resources import CDN
from dask import bag as dbag
from multiprocessing.pool import ThreadPool
np.random.seed(2018)
path_bag = dbag.from_sequence(np.random.choice(all_series, 170))
image_bag = path_bag.map(read_and_process)
with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof, dask.set_options(pool = ThreadPool(4)):
    all_img_data = image_bag.compute()
output_notebook(CDN, hide_banner=True)
diag.visualize([prof, rprof])
im_stack = np.concatenate([x[-1] for x in all_img_data if x is not None],0)
print(im_stack.shape)
am_stack = np.concatenate([ [x[2]]*x[-1].shape[0] for x in all_img_data if x is not None],0)
print(am_stack.shape)
path_stack = np.concatenate([ [os.path.basename(x[0])]*x[-1].shape[0] for x in all_img_data if x is not None],0)
print(path_stack.shape)
time_stack = np.concatenate([ [x[1]]*x[-1].shape[0] for x in all_img_data if x is not None],0)
print(time_stack.shape)
import pandas as pd
train_targets = {int(k['Id']): k for k in pd.read_csv('../input/train.csv').T.to_dict().values()}
import h5py
with h5py.File('train_mri_{}_{}.h5'.format(X_DIM, Y_DIM), 'w') as w:
    w.create_dataset('image', data = im_stack, compression = 9)
    w.create_dataset('systole', data = [train_targets[int(c_id)]['Systole'] for c_id in path_stack])
    w.create_dataset('diastole', data = [train_targets[int(c_id)]['Diastole'] for c_id in path_stack])
    w.create_dataset('id', data = [int(c_id) for c_id in path_stack])
    w.create_dataset('area_multiplier', data = am_stack)
    w.create_dataset('time', data = time_stack)
!ls -lh *
