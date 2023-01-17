!pip install -q easyocr
import os



import numpy as np

import pandas as pd

from glob import glob

from time import time

from tqdm.auto import tqdm 



from PIL import Image



import easyocr

import Levenshtein
ROOT_DIR = "../input/mlub-mongolian-car-plate-prediction"
df_train = pd.read_csv(os.path.join(ROOT_DIR, "training.csv"))

df_subm  = pd.read_csv(os.path.join(ROOT_DIR, "submission.csv"))

print(df_train.shape, df_subm.shape)
df_train.sample(5)
%%time

reader = easyocr.Reader(['mn'])
image_path = f"{ROOT_DIR}/training/training/0002.png"

Image.open(image_path)
# %%time

result = reader.readtext(image_path, detail=0)

result = "".join(result)[:7]

result
# %%time

p1 = '5791УБВ'

p2 = '5793882'



print(f"{p1}&{p2} => levenstein {Levenshtein.distance(p1, p2)}")
test_paths = sorted(glob(os.path.join(ROOT_DIR, "test/test/*")))

len(test_paths)
ids     = []

y_preds = []



for tp in tqdm(test_paths):

    result = reader.readtext(tp, detail=0)

    result = "".join([a.strip() for a in result])[:7]

    

    ids.append(tp.split("/")[-1].split(".")[0])

    y_preds.append(result)
df_subm["plate_number"] = y_preds
df_subm.to_csv("submission.csv", index=False)