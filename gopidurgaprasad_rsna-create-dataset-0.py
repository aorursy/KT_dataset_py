import os, glob
import numpy as np
import pandas as pd
import shutil
import joblib

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
train_df = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")
train_df.head()
gkf = GroupKFold(n_splits=4)

train_df["kfold"] = -1

y = train_df["StudyInstanceUID"].values

for i, (train_idx, valid_idx) in enumerate(gkf.split(train_df, train_df, groups=y)):
    train_df.loc[valid_idx, "kfold"] = i

train_df["kfold"].value_counts()
train_fold3 = train_df[train_df.kfold == 3].copy()
train_fold3.shape
!mkdir -p /root/.kaggle/
!cp ../input/mykaggleapi/kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!mkdir -p "/tmp/RSNA"
SAVE_PATH = "/tmp/RSNA/train_3"
def create_dataset(file_name1, file_name2):
    
    image_paths = glob.glob(f"../input/rsna-str-pe-detection-jpeg-256/train-jpegs/{file_name1}/{file_name2}/*.jpg")
    
    save_path = f'{SAVE_PATH}/{file_name1}/{file_name2}'
    os.makedirs(save_path, exist_ok=True)
    
    for f in image_paths:
        shutil.copy(f, save_path)
drop_df = train_fold3[['StudyInstanceUID', 'SeriesInstanceUID']]
drop_df = drop_df.drop_duplicates()
drop_df.shape
file_paths = drop_df[['StudyInstanceUID', 'SeriesInstanceUID']].values.tolist()
len(file_paths)
_ = Parallel(n_jobs=8, backend="multiprocessing")(
    delayed(create_dataset)(path[0], path[1]) for path in tqdm(file_paths, total=len(file_paths))
)
!ls "/tmp/RSNA/"
train_fold3.to_csv("/tmp/RSNA/train_fold3.csv", index=False)
!ls "/tmp/RSNA/"
!zip -r "/tmp/RSNA/train_3.zip" "/tmp/RSNA/train_3" >> quit
!ls -l "/tmp/RSNA/"
data = '''{
  "title": "rsna-str-fold3-jpeg-256",
  "id": "gopidurgaprasad/rsna-str-fold3-jpeg-256",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
'''
text_file = open("/tmp/RSNA/dataset-metadata.json", 'w+')
n = text_file.write(data)
text_file.close()
!ls -l "/tmp/RSNA/"
!kaggle datasets create -p "/tmp/RSNA/"
