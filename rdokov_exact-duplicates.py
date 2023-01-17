import numpy as np

import pandas as pd

import os

import hashlib



# first we collect some basic information about the images

records = []

for name in os.listdir('../input/train/'):

    if 'mask' in name or not name.endswith('.tif'):

        continue



    patient_id, image_id = name.strip('.tif').split('_')

    with open('../input/train/' + name, 'rb') as fd:

        md5sum = hashlib.md5(fd.read()).hexdigest()



    records.append({

        'filename': name,

        'patient_id': patient_id,

        'image_id': image_id,

        'md5sum': md5sum,

    })



df = pd.DataFrame.from_records(records)
# select the files that occur more than once

counts = df.groupby('md5sum')['filename'].count()

duplicates = counts[counts > 1]

print(len(duplicates))
# some files occur more than twice

duplicates[duplicates > 2]
# there also appears to be some strange mixup between patients 17 & 18

for md5sum in duplicates.index:

    subset = df[df['md5sum'] == md5sum]

    if len(subset['patient_id'].value_counts()) > 1:

        print(subset)

        print('------')