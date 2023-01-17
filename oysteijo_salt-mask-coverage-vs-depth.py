import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
from keras.preprocessing.image import load_img
from tqdm import tqdm_notebook
img_size_ori = 101
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), color_mode="grayscale")) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / (img_size_ori * img_size_ori)
plt.figure(figsize=(15,12))
sns.scatterplot(x="z", y="coverage", data=train_df)
np.corrcoef(train_df.z , train_df.coverage)[0,1]