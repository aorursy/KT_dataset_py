# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#This is not a submission - it's a test, how to make a submission !!!

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







for petid1, petid2 in sorted(list(dups)):

    row1 = detail[petid1]

    row2 = detail[petid2]

    if row1.Category != row2.Category:

        show(row1, row2)