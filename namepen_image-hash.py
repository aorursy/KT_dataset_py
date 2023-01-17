import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

from os.path import isfile

import pickle

from tqdm import tqdm_notebook

print(os.listdir("../input"))



from PIL import Image as pil_image

import matplotlib.pyplot as plt



from imagehash import phash #img의 P-hash 값을 구하는 라이브러리

train_df = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/train.csv')

test_df = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/test.csv')

sample_df = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/sample_submission.csv')



basic_path = '../input/2019-3rd-ml-month-with-kakr/'

save_path = '../input/hashdataset/'
tagged = dict([(p,w) for _, p,w in train_df[['img_file', 'class']].to_records()]) # img_file : class_num, dict

submit = [p for _, p in sample_df[['img_file']].to_records()] #img_file, dict

join = list(tagged.keys()) + submit #img_file list tagged and submit
#img_file을 불러오는 함수

def expand_path(p):    

    if 'train' in p:

        p = basic_path +'train/'+ p

    else:

        p = basic_path + 'test/'+ p

    return p
def match(h1, h2):

    for p1 in h2ps[h1]:

        for p2 in h2ps[h2]:

            i1 = pil_image.open(expand_path(p))

            i2 = pil_image.open(expand_path(p))

            

            if i1.mode != i2.mode or i1.size != i2.size: #mode is RGB or grayscale

                return False

            

            a1 = np.array(i1)

            a1 = a1-a1.mean()

            a1 = a1/sqrt((a1**2).mean())

            

            a2 = np.array(i2)

            a2 = a2-a2.mean()

            a2 = a2/sqrt((a2**2).mean())

            

            a = ((a1-a2)**2).mean()

            if a > 0.1:

                return False

    return True
'''#이미지의 해시 값이 특정 조건을 만족하면, 같은 해시값으로 처리

h2h = {}



for i, h1 in enumerate(tqdm_notebook(hs)):

    for h2 in hs[:i]:

        if h1-h2 <= 6 and match(h1,h2):

            s1 = str(h1)

            s2 = str(h2)

            if s1 < s2:

                s1,s2 = s2,s1

            h2h[s1] = s2

            

with open('h2h.pickle', 'wb') as handle:

    pickle.dump(h2h, handle, protocol=pickle.HIGHEST_PROTOCOL)'''
with open(save_path + 'p2h.pickle', 'rb') as f:

    p2h = pickle.load(f)

    

with open(save_path + 'h2h.pickle', 'rb') as f:

    h2h = pickle.load(f)
#phash 값이 같은 이미지 찾기

h2ps = {}



for p,h in p2h.items():

    if h not in h2ps:

        h2ps[h] = []

    if p not in h2ps[h]:

        h2ps[h].append(p)
#h2h에서 같은 값으로 처리한 이미지의 hash값을 같도록 변경.

for p,h in p2h.items():

    h = str(h)

    if h in h2h:

        h = h2h[h]

    p2h[p] = h
#총 45개가 차이 발생

print(len(h2ps), len(p2h))



cc = 0

for h, ps in h2ps.items():

    if len(ps) >= 2:

        cc +=1

        

print(cc)
c = 0

for h, ps in h2ps.items():

    if len(ps) >= 2:

        c+=1

        print('Images : ', ps)

        g = []

        for p in ps:

            if 'test' in p:

                g.append(' ')

                continue

            g.append(tagged[p])

        print('Class : ', g)

        

print("Total : ", c)
def show_car(imgs, per_row=2):

    n = len(imgs)

    rows = (n + per_row - 1)//per_row

    cols = min(per_row, n)

    fig, axes = plt.subplots(rows,cols, figsize=(24//per_row*cols,24//per_row*rows))

    for ax in axes.flatten(): ax.axis('off')

    for i,(img,ax) in enumerate(zip(imgs, axes.flatten())): 

        ax.imshow(img.convert('RGB'))
for h, ps in h2ps.items():

    if len(ps) >= 2:

        #print('Images:', ps)

        imgs = [pil_image.open(expand_path(p)) for p in ps]

        show_car(imgs, per_row=len(ps))