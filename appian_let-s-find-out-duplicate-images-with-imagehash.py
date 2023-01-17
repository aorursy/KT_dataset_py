%load_ext autoreload

%autoreload 2

%matplotlib inline



import glob

import itertools

import collections



from PIL import Image

import cv2

from tqdm import tqdm_notebook as tqdm

import pandas as pd

import numpy as np

import torch

import imagehash



import matplotlib.pyplot as plt
def run():



    funcs = [

        imagehash.average_hash,

        imagehash.phash,

        imagehash.dhash,

        imagehash.whash,

        #lambda x: imagehash.whash(x, mode='db4'),

    ]



    petids = []

    hashes = []

    for path in tqdm(glob.glob('../input/*_images/*-1.jpg')):



        image = Image.open(path)

        imageid = path.split('/')[-1].split('.')[0][:-2]



        petids.append(imageid)

        hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))



    return petids, np.array(hashes)



%time petids, hashes_all = run()
hashes_all = torch.Tensor(hashes_all.astype(int)).cuda()
%time sims = np.array([(hashes_all[i] == hashes_all).sum(dim=1).cpu().numpy()/256 for i in range(hashes_all.shape[0])])
indices1 = np.where(sims > 0.9)

indices2 = np.where(indices1[0] != indices1[1])

petids1 = [petids[i] for i in indices1[0][indices2]]

petids2 = [petids[i] for i in indices1[1][indices2]]

dups = {tuple(sorted([petid1,petid2])):True for petid1, petid2 in zip(petids1, petids2)}

print('found %d duplicates' % len(dups))
train = pd.read_csv('../input/train/train.csv')

test = pd.read_csv('../input/test/test.csv')



train.loc[:,'Category'] = 'train'

test.loc[:,'Category'] = 'test'

test.loc[:,'AdoptionSpeed'] = np.nan



df = pd.concat([train, test], sort=False)
detail = {petid:df[df.PetID == petid].iloc[0] for petid in itertools.chain.from_iterable(list(dups))}
def show(row1, row2):



    print('PetID: %s / %s' % (row1.PetID, row2.PetID))

    print('Name: %s / %s' % (row1.Name, row2.Name))

    print('Category: %s / %s' % (row1.Category, row2.Category))

    print('AdoptionSpeed: %s / %s' % (row1.AdoptionSpeed, row2.AdoptionSpeed))

    print('Breed1: %d / %d' % (row1.Breed1, row2.Breed1))

    print('Age: %d / %d' % (row1.Age, row2.Age))

    print('RescuerID:\n%s\n%s' % (row1.RescuerID, row2.RescuerID))

    

    image1 = cv2.imread('../input/%s_images/%s-1.jpg' % (row1.Category, row1.PetID))

    image2 = cv2.imread('../input/%s_images/%s-1.jpg' % (row2.Category, row2.PetID))

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    

    fig = plt.figure(figsize=(10, 20))

    fig.add_subplot(1,2,1)

    plt.imshow(image1)

    fig.add_subplot(1,2, 2)

    plt.imshow(image2)

    plt.show()
for petid1, petid2 in sorted(list(dups)):

    row1 = detail[petid1]

    row2 = detail[petid2]

    if row1.Category != row2.Category:

        show(row1, row2)
counter = collections.Counter()

for petid1, petid2 in list(dups):

    row1 = detail[petid1]

    row2 = detail[petid2]

    

    for attr in train.columns:

        if getattr(row1, attr) != getattr(row2, attr):

            counter[attr] += 1

            

counter
for petid1, petid2 in list(dups)[:20]:

    row1 = detail[petid1]

    row2 = detail[petid2]

    if row1.Description != row2.Description:

        print(row1.Description)

        print('-'*5)

        print(row2.Description)

        print('\n')
import json

out = [[petid1,petid2] for petid1,petid2 in dups.keys()]

with open('dups.json', 'w') as fp:

    fp.write(json.dumps(out))