import os
from zipfile import ZipFile
import cv2
import numpy as np
import pandas as pd
from dask import bag, threaded
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
import imagehash
# get filenames
zipped = ZipFile('../input/test_jpg.zip')
filenames = zipped.namelist()[1:] # exclude the initial directory listing
print(len(filenames))
#filenames = filenames[0:10]
# define function
def get_hashes2(file):
    exfile = zipped.read(file)
    arr = np.frombuffer(exfile, np.uint8)
    if arr.size > 0:   # exclude dirs and blanks
        imz = cv2.imdecode(arr, flags=cv2.IMREAD_GRAYSCALE)
        shp = imz.shape
    else: 
        shp = [0,0,0] 
    return (file, shp)

exfile = zipped.read(filenames[1])
arr = np.frombuffer(exfile, np.uint8)
imz = cv2.imdecode(arr, flags=cv2.IMREAD_GRAYSCALE)
shp = imz.shape
shp

file = [zinfo.filename for zinfo in  zipped.filelist[1:]]
file[-5:]
size_train = [zinfo.file_size//1000 for zinfo in  zipped.filelist[1:]]
size_train[-5:]
compress_size_train =[zinfo.compress_size/1000 for zinfo in  zipped.filelist[1:]]
compress_size_train[-5:]
out = pd.DataFrame({'image':file, 'size':size_train,  'csize':compress_size_train})
out['image'] = out.image.str.replace("data/competition_files/test_jpg/", "")
out['image'] = out.image.str.replace(".jpg", "")
out.sort_values('image').head()
out.to_csv('test_img_files_info.csv')
out.head()
# define function
def get_hashes3(file):
    exfile = zipped.read(file)
    arr = np.frombuffer(exfile, np.uint8)
    if arr.size > 0:   # exclude dirs and blanks
        img = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
        average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
        average_color
    else: 
        average_color = [0,0,0] 
    return (file, average_color)
exfile = zipped.read(filenames[1])
arr = np.frombuffer(exfile, np.uint8)
img = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
average_color

b = bag.from_sequence(filenames).map(get_hashes2)
with ProgressBar():
    h_list_trn = b.compute(get=threaded.get)
print(h_list_trn[0:10])
b = bag.from_sequence(filenames).map(get_hashes3)
with ProgressBar():
    c_list_trn = b.compute(get=threaded.get)
print(c_list_trn[0:10])
pd.options.display.max_colwidth = 100   # show the annoyingly long strings
names_ = [h[0] for h in h_list_trn]
dim_ = [h[1] for h in h_list_trn]
col_ = [h[1] for h in c_list_trn]

hash_df_trn = pd.DataFrame({'image':names_, 'dim':dim_,  'colors':col_})
hash_df_trn['image'] = hash_df_trn.image.str.replace("data/competition_files/test_jpg/", "")
hash_df_trn['image'] = hash_df_trn.image.str.replace(".jpg", "")
hash_df_trn.sort_values('image').head()
hash_df_trn.to_csv('train_img_feat.csv')
hash_df_trn.head()