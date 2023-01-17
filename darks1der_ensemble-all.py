import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

files = glob.glob("/*/*/*/*.npy")

preds = 0

for f in files:

    preds+=np.load(f)

preds/=4
train = pd.read_csv('../input/clabscvcomp/data/train.csv')
preds = np.argmax(preds,axis = 1)

categories = sorted(train.genres.unique().astype('str'))

final_preds = []

for idx in preds:

    final_preds.append(categories[idx])

final_submit = pd.read_csv('../input/clabscvcomp/data/sample_submission.csv')

final_submit.genres = final_preds

final_submit.head()

final_submit.to_csv('submission.csv',index = False)