# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! pip -q install fastai2 nbdev

from fastai2.text.all import *
data_drive="/kaggle/input/nlp-getting-started/"

competition_google_drive="/kaggle/input/nlp-getting-started/"

# Loading the data

dls = TextDataLoaders.from_csv(data_drive, csv_fname="train.csv",valid_pct=0.1,text_col="text",label_col="target")

# loading the classifier

f_score=F1Score()

learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=[accuracy,f_score],path=competition_google_drive)

learn.fine_tune(6, 1e-2)
test_items=pd.read_csv(data_drive + "test.csv")

dl = learn.dls.test_dl(test_items)

preds, _ , classif = learn.get_preds(dl=dl,with_decoded=True)
ss=pd.read_csv(data_drive+"sample_submission.csv")

ss.head()

ss['id']=dl.get_idxs()

ss['target']=classif

ss=ss.sort_values(by=['id'])

ss=ss.reset_index(drop=True)

ss['id']=test_items['id']

ss.to_csv('submission.csv', index=False)
