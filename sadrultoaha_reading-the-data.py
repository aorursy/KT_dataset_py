import os

import pandas as pd

import numpy as np

import PIL.Image

import cv2

import pyarrow.parquet as pq
df = pd.read_parquet('/kaggle/input/bengaliai-ocr-2019/train_image_data_0.parquet')



table2 = pq.read_table('/kaggle/input/bengaliai-ocr-2019/train_image_data_0.parquet')

table2.to_pandas()
df.shape
flattened_image = df.iloc[4].drop('image_id').values.astype(np.uint8)
unpacked_image = PIL.Image.fromarray(flattened_image.reshape(137, 236))
unpacked_image