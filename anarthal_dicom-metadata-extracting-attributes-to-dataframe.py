import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydicom

from tqdm import tqdm

import os
dcms = []

for root, dirs, fnames in os.walk('/kaggle/input/osic-pulmonary-fibrosis-progression/train'):

    dcms += list(os.path.join(root, f) for f in fnames if f.endswith('.dcm'))

print(f'There are {len(dcms)} CT scans')
attrs = set()

for fname in tqdm(dcms):

    with pydicom.dcmread(fname) as obj:

        attrs.update(obj.dir())
dcm_keys = list(attrs)

dcm_keys.remove('PixelData') # The actual array of pixels, this is not metadata

dcm_keys.remove('PatientName') # Anonymous data!

dcm_keys
meta = []

typemap = {

    pydicom.uid.UID: str,

    pydicom.multival.MultiValue: list

}

def cast(x):

    return typemap.get(type(x), lambda x: x)(x)



for i, fname in enumerate(tqdm(dcms)):

    with pydicom.dcmread(fname) as obj:

        meta.append([cast(obj.get(key, np.nan)) for key in dcm_keys])



dfmeta = pd.DataFrame(meta, columns=dcm_keys)

dfmeta
dfmeta.to_csv('meta.csv', index=False)