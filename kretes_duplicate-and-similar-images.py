import os

import pandas as pd

import numpy as np
import imagehash

from PIL import Image

def img_hash(file, hash_size):    

    return imagehash.phash(Image.open(file),hash_size=hash_size)
def get_hashes(directory, hash_size):

    hash_file = 'img_hashes_%s.csv' % hash_size

    if not os.path.isfile(hash_file):

        hashes = pd.DataFrame()

    else:

        hashes = pd.read_csv(hash_file)

    new_hashes_calculated = 0

    num_of_files=len(os.listdir(directory))

    for file in os.listdir(directory):

        if 'file' not in hashes.columns or file not in list(hashes['file']):                                               

            new_hashes_calculated = new_hashes_calculated + 1

            result = {'file': file,'hash':img_hash(directory + '/' + file,hash_size)}

            hashes = hashes.append(result,ignore_index=True)

            if (new_hashes_calculated % 200 == 199):

                hashes[['file','hash']].to_csv(hash_file,index=False) 

    if new_hashes_calculated:

        hashes[['file','hash']].to_csv(hash_file,index=False)    

    return read_hashes(hash_size)
def read_hashes(hash_size):

    hash_file = 'img_hashes_%s.csv' % hash_size

    hashes = pd.read_csv(hash_file)[['file','hash']]

    lambdafunc = lambda x: pd.Series([int(i,16) for key,i in zip(range(0,len(x['hash'])),x['hash'])])

    newcols = hashes.apply(lambdafunc, axis=1)

    newcols.columns = [str(i) for i in range(0,len(hashes.iloc[0]['hash']))]

    return hashes.join(newcols) 
hashes_4 = get_hashes('../input/test_stg1',4)
hashes_4.head()
#are there any duplicates in terms of hashes of size 4?

print("%s out of %s" % (len(hashes_4[hashes_4.duplicated(subset='hash',keep=False)]),len(hashes_4)))
hashes_16_lag = get_hashes('../input/train/LAG/',16)
hashes_16_lag.head()
#are there any duplicates in terms of hashes of size 16?

len(hashes_16_lag[hashes_16_lag.duplicated(subset='hash',keep=False)])
from sklearn.neighbors import KDTree

t = KDTree(hashes_16_lag[[str(i) for i in range(0,64)]],metric='manhattan')
distances, indices = t.query(hashes_16_lag[[str(i) for i in range(0,64)]],k=2)
distances[:,1]
index_of_closest_distances = np.argsort(distances[:,1])
distances[index_of_closest_distances[:10]]
indices_pairs_of_closest_distance = indices[index_of_closest_distances]

indices_pairs_of_closest_distance[:10]
unique_pairs = [pair for pair in indices_pairs_of_closest_distance if (pair == np.sort(pair)).all()]

unique_pairs[:5]
hashes_16_lag.iloc[unique_pairs[0]]
len(unique_pairs)
def read_image_bytes(filename):

    with open(filename, mode='rb') as file:

        return file.read()

    

def read_image_numpy(filename, w, h):

    from PIL import Image

    from numpy import array

    img = Image.open(filename).resize((w,h))

    img = img.convert('RGB')

    return array(img)



def scale(arr):

    return arr / 255.0



import matplotlib.pyplot as plt

%matplotlib inline

def mosaic_images(images_tensor, ncols, grayscale=False):

    img_size = images_tensor.shape[1]

    col_size = ncols*(img_size +1) -1

    nrows = int(np.ceil(images_tensor.shape[0] / ncols))

    row_size = nrows*(img_size +1)-1

    if grayscale:

        final = np.ones((row_size,col_size))

    else:

        final = np.ones((row_size,col_size,3))

    

    for i in range(images_tensor.shape[0]):

        row = int(np.floor(i / ncols))

        col = i % ncols

        kernel = images_tensor[i]

        x = col * (img_size+1)

        y = row * (img_size+1)

        final[y:y+img_size,x:x+img_size] = kernel

    return final
fromm = 0

too = 10

file_names = hashes_16_lag.iloc[np.ndarray.flatten(np.array(unique_pairs[fromm:too]))]

files_to_show = [scale(read_image_numpy('../input/train/LAG/%s' % f,400,400)) for f in list(file_names['file'])]

len(files_to_show)
## plt.figure(figsize=(10,how_many*5))

## plt.imshow(mosaic_images(np.asarray(files_to_show),2))



# You must fork this notebook to the private space since we cannot show images from the competition in public kernels
hashes_sample = hashes_16_lag.sample(n=10,random_state = 124)

distances_10, indices_10 = t.query(hashes_sample[[str(i) for i in range(0,64)]],k=18)
distances_10
print(distances_10[0])

other_images = list(hashes_16_lag.iloc[indices_10[4]]['file'])

images = [scale(read_image_numpy('../input/train/LAG/%s' % file,500,500)) for file in other_images]

## plt.figure(figsize=(10,20))

## plt.imshow(mosaic_images(np.array(images),3))

# base image is in the top left corner
# now let's look at some image without so close neighbours

print(distances_10[5])

other_images = list(hashes_16_lag.iloc[indices_10[5]]['file'])

images = [scale(read_image_numpy('../input/train/LAG/%s' % file,500,500)) for file in other_images]

## plt.figure(figsize=(10,20))

## plt.imshow(mosaic_images(np.array(images),3))



# base image is in the top left corner
from sklearn.neighbors import KDTree



def duplicates_in_dir(directory, hash_size,threshold, return_original=None):

    hashes = get_hashes(directory,hash_size)

    hash_str_len = len(hashes.get_value(0,'hash'))

    files_in_dir = os.listdir(directory)

    hashes = hashes[hashes['file'].isin(files_in_dir)]    

    print('calculating distances')

    t = KDTree(hashes[[str(i) for i in range(0,hash_str_len)]],metric='manhattan')

    distances, indices = t.query(hashes[[str(i) for i in range(0,hash_str_len)]],k=5)

    above_threshold_idx = np.argwhere((distances<=threshold) & (distances>0))

    pairs_of_indexes_ofduplicate_images = set([tuple(sorted([indices[idx[0],0],indices[idx[0],idx[1]]])) for idx in above_threshold_idx])

    to_remove = [t[1] for t in pairs_of_indexes_ofduplicate_images]

    files_to_remove = [os.path.join(directory,f) for f in list(hashes.iloc[to_remove]['file'])]

    if return_original:

        to_keep = [t[0] for t in pairs_of_indexes_ofduplicate_images]

        files_to_keep = [os.path.join(directory,f) for f in list(hashes.iloc[to_keep]['file'])]

        return (files_to_keep, files_to_remove)

    else:

        return files_to_remove
def get_duplicate_report():

    for clazz in os.listdir('../input/train'):

        if clazz != '.DS_Store':

            base_dir = '../input/train'

            to_keep, to_remove = duplicates_in_dir('%s/%s' % (base_dir,clazz),16,10,True)

            yield list(zip(to_keep, to_remove))
r = list(get_duplicate_report())
def flatten(lists):

        return [elem for lis in lists for elem in lis]

total_report = flatten(r)
print('we have found %s duplicates in train set' % len(total_report))
for i in range(0,12):

    chunk = total_report[i*10:(i+1)*10]

    images = [scale(read_image_numpy(c[j],400,400)) for c in chunk for j in [0,1]]

    ## plt.figure(figsize=(8,40))

    ## plt.imshow(mosaic_images(np.array(images),2))
dups_df = pd.DataFrame(total_report)

dups_df.columns = ['keep','remove']

dups_df['hash_size'] = 16

dups_df['threshold'] = 10

dups_df.to_csv('dups_hash16_dist10.csv',index=False)

dups_df.head()
to_keep, to_remove = duplicates_in_dir('../input/test_stg1/',16,10,True)

test_dups = list(zip(to_keep,to_remove))
dups_test_df = pd.DataFrame(test_dups)

dups_test_df.columns = ['keep','remove']

dups_test_df.to_csv('dups_test_hash16_dist10.csv',index=False)

dups_test_df.head()