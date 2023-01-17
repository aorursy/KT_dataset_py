import cv2
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
TRAIN = '../input/train_v2/'
names = os.listdir(TRAIN)
def get_hash(img):
    result = []
    sz = 256 #images are composed of 256x256 overlapping patches
    for ix in range(0,768,sz):
        for jx in range(0,768,sz):
            result.append(hash(img[ix:ix+sz,jx:jx+sz,:].tobytes()))
    return result
hash_dict = defaultdict(list)
for name in tqdm(names):
    img = cv2.imread(os.path.join(TRAIN,name))
    hashes = get_hash(img)
    for h in hashes:
        hash_dict[h].append(name)
img_dict = {name:list() for name in names}
for key,val in hash_dict.items():
    items = set(val)
    if(len(items) > 1):
        for item_i in items:
            for item_j in items:
                if item_i == item_j: continue
                img_dict[item_i].append(item_j)
                
duplicate_dict = dict()
for key,val in img_dict.items():
    duplicates = set(val)
    if len(duplicates) == 0: duplicate_dict[key] = np.nan
    else: duplicate_dict[key] = ' '.join(duplicates)
df = pd.DataFrame(pd.Series(duplicate_dict), columns=['duplicates'])
df['ImageId'] = df.index
df.to_csv('duplicates.csv',index=False)
df.head()