# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import pydicom
import matplotlib.patches as patches
from skimage.filters import *

# Any results you write to the current directory are saved as output.
def my_list(rows):
    for row in rows:
        return list(row)
df = pd.read_csv("../input/stage_1_train_labels.csv")
#                 converters={"x" : np.float64, "y" : np.float64, "width" : np.float64, "height" : np.float64})
df.head()
df_patient = pd.DataFrame(df.groupby("patientId")["x"].apply(list))
df_patient["y"] = pd.DataFrame(df.groupby("patientId")["y"].apply(list))
df_patient["width"] = pd.DataFrame(df.groupby("patientId")["width"].apply(list))
df_patient["height"] = pd.DataFrame(df.groupby("patientId")["height"].apply(list))
df_patient["Target"] = pd.DataFrame(df.groupby("patientId")["Target"].apply(list))

def display_image(row, col, img, values, title):
    ax[row][col].set_title(title)
    ax[row][col].imshow(img, cmap=plt.cm.bone) 
    for val in values:
        for i in range(len(val[0])):
            rect = patches.Rectangle((val[0][i], val[1][i]),
                                     val[2][i], val[3][i],linewidth=1,edgecolor='r',facecolor='none')
            ax[row][col].add_patch(rect)

last_target = 1
count = 0
num_cols = 2
fig,ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 20))
while count < 6:
    sample = df_patient.sample()
    if  sample.iloc[0]["Target"][0] == last_target:
        continue
    last_target = 0 if last_target == 1 else 1
    filename =  "../input/stage_1_train_images/" + sample.index[0] + ".dcm"
    ds = pydicom.dcmread(filename)
    row = int(count/num_cols)
    col = count%num_cols
    display_image(row, col,ds.pixel_array,sample.values, sample.index[0]  )
    count += 1

plt.show()

# def display_image(row, col, ax, image, values):
#     ax[row][col].imshow(image, cmap=plt.cm.bone) 
#     for val in values:
#         for i in range(len(val[0])):
#             rect = patches.Rectangle((val[0][i], val[1][i]),
#                                      val[2][i], val[3][i],linewidth=1,edgecolor='r',facecolor='none')
#             ax[row][col].add_patch(rect)
  

count = 0
num_cols = 2
fig,ax = plt.subplots(nrows=4, ncols=2, figsize=(8, 20))
while count < 6:
    sample = df_patient.sample()
    filename =  "../input/stage_1_train_images/" + sample.index[0] + ".dcm"
    ds = pydicom.dcmread(filename)
    row = int(count/num_cols)
    col = count%num_cols
    img = ds.pixel_array
    display_image(row, col, img, sample.values, "original" )
    count += 1
    from skimage import exposure
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    row = int(count/num_cols)
    col = count%num_cols
    display_image(row, col, img_rescale, sample.values, "rescale" )
    count += 1
    row = int(count/num_cols)
    col = count%num_cols
    display_image(row, col, img_eq, sample.values, "hist eq" )
    count += 1
    row = int(count/num_cols)
    col = count%num_cols
    display_image(row, col, img_adapteq, sample.values, "adaptive eq" )
    count += 1
plt.show()
