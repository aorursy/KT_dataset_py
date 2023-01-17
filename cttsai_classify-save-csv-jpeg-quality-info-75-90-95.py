!apt-get update

!apt -y install imagemagick
from typing import List

from glob import glob

import os

import pandas as pd



working_dir: str = "../input/alaska2-image-steganalysis/"



    

training_images: List[str] = list(glob(os.path.join(working_dir, "*", "*.jpg")))

df_train = pd.DataFrame({"file_path": training_images})

df_train["image"] = df_train["file_path"].apply(lambda x: os.path.basename(x))

df_train["kind"] = df_train["file_path"].apply(lambda x: os.path.split(os.path.dirname(x))[-1])

df_train
from typing import Optional



def func(file_path: str) -> Optional[float]:

    output = os.popen(f"identify -format '%Q' {file_path}").read()

    return output

from multiprocessing import Pool

with Pool(4) as p:

    ret = list(p.map(func, training_images))
df_train["quality"] = ret

df_train["quality"] = df_train["quality"].astype("float")

df_train.sort_values("image", inplace=True)

df_train[["image", "kind", "quality"]].to_csv("image_quality.csv", index=False, float_format='%.0f')
df_train["quality"].value_counts(normalize=False)