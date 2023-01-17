import glob

import joblib

import pickle

import pydicom

import pandas as pd

from tqdm.notebook import tqdm
dicom_files = glob.glob('../input/rsna-str-pulmonary-embolism-detection/train/*/*/*')

print( len(dicom_files) )

print( dicom_files[0] )
def read_meta(fn):

    dc = pydicom.dcmread(fn, stop_before_pixels=True )

    RES = {}

    for i in list(dc.keys()):

        RES[dc[i].description()] = dc[i].value

    return RES



#Read 1% of train files

RES = joblib.Parallel(n_jobs=2)(joblib.delayed(read_meta)(fn) for fn in tqdm(dicom_files[:int(0.01*len(dicom_files))]))
print(len(RES))

print(RES[0])
df = pd.DataFrame(RES)

df.sample(10).head(10)
df.sample(10).head(10)
df.to_pickle('train-metadata.pkl')