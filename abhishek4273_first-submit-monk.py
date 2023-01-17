import os

import sys
os.listdir("/kaggle/input/workspace/workspace")
! cp -r /kaggle/input/workspace/workspace .
os.listdir("./")
os.listdir("/kaggle/input/bengaliai-cv19/")
os.listdir("/kaggle/input/monk-kaggle-bengali-ai/monk_kaggle_bengali_ai/")
!cd /kaggle/input/monk-kaggle-bengali-ai/monk_kaggle_bengali_ai/installs/ && pip install PyLg-1.3.3-py3-none-any.whl
!cd /kaggle/input/monk-kaggle-bengali-ai/monk_kaggle_bengali_ai/installs/ && pip install blessings-1.7-py3-none-any.whl
!cd /kaggle/input/monk-kaggle-bengali-ai/monk_kaggle_bengali_ai/installs/ && pip install netron-3.7.3-py2.py3-none-any.whl
!cd /kaggle/input/monk-kaggle-bengali-ai/monk_kaggle_bengali_ai/installs/ && pip install gpustat-0.6.0-py3-none-any.whl
import sys

sys.path.append("/kaggle/input/monk-kaggle-bengali-ai/monk_kaggle_bengali_ai/monk/")

sys.path.append("/kaggle/input/monk-kaggle-bengali-ai/monk_kaggle_bengali_ai/installs/")
from gluon_prototype import prototype
from tqdm.notebook import tqdm

import pandas as pd



import numpy as np

import mxnet as mx
gtf_list = []



gtf1 = prototype(verbose=1);

gtf1.Prototype("sample-project", "sample-experiment-1", eval_infer=True);

gtf_list.append(gtf1);





gtf2 = prototype(verbose=1);

gtf2.Prototype("sample-project", "sample-experiment-2", eval_infer=True);

gtf_list.append(gtf2);



gtf3 = prototype(verbose=1);

gtf3.Prototype("sample-project", "sample-experiment-3", eval_infer=True);

gtf_list.append(gtf3);
target = ["consonant_diacritic", "grapheme_root", "vowel_diacritic"];

combined = [];





for i in range(4):

    fname = "/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet".format(i);

    print("reading - ", fname);

    df = pd.read_parquet(fname);

    

    for j in range(len(df)):

        image_id = df.iloc[j][0];

        #print(image_id);

        data = df.iloc[j][1:];



        for k in range(3):

            id_ = image_id + "_" +  target[k]

            predictions = gtf_list[k].Infer_Kaggle(data)

            pred = int(predictions["predicted_class"]);

            #print(id_, pred)

            combined.append([id_, pred]);
df = pd.DataFrame(combined, columns = ['row_id', 'target']);  
df.to_csv("submission.csv", index=False)
df[:5]
! rm -r workspace