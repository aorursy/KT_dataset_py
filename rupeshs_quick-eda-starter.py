import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

train_df=pd.read_csv("../input/train.csv")
train_df.head()
train_df.info()
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open("../input/train/0000e88ab.jpg")

plt.imshow(im)
im.size
whales = train_df.groupby('Id')['Image'].nunique()
whales.sort_values(ascending=False)

whales.sort_values(ascending=False)[1:100].hist()
test_df=pd.read_csv("../input/sample_submission.csv")
test_df.head()
im_test = Image.open("../input/test/001a4d292.jpg")
plt.imshow(im_test)