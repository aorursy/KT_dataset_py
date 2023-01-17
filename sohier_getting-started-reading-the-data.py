import os

import pandas as pd

import numpy as np

import PIL.Image
df = pd.read_parquet('/kaggle/input/bengaliai-ocr-2019/train_image_data_1.parquet')
df.shape
flattened_image = df.iloc[123].drop('image_id').values.astype(np.uint8)
unpacked_image = PIL.Image.fromarray(flattened_image.reshape(137, 236))
unpacked_image