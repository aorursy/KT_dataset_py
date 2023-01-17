# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv("../input/global-wheat-detection/train.csv")

train_data.head()
train_data["area"]=[float(eval(train_data["bbox"][i])[2])* float(eval(train_data["bbox"][i])[3]) for i in range(len(train_data))]
train_data.head()
sorted_df=train_data.sort_values(by=["area"]).reset_index(drop=True)

sorted_df["bbox"]=[eval(i) for i in sorted_df["bbox"]]

sorted_df["bbox_xmin"]=[float(sorted_df["bbox"][i][0]) for i in range(len(sorted_df))]

sorted_df["bbox_ymin"]=[float(sorted_df["bbox"][i][1]) for i in range(len(sorted_df))]

sorted_df["bbox_width"]=[float(sorted_df["bbox"][i][2]) for i in range(len(sorted_df))]

sorted_df["bbox_height"]=[float(sorted_df["bbox"][i][3]) for i in range(len(sorted_df))]
from PIL import Image

import matplotlib.pyplot as plt

import matplotlib.patches as patches



TRAIN_DIR="../input/global-wheat-detection/train/"





def get_all_bboxes(df, image_id):

    image_bboxes = df[df.image_id == image_id]

    

    bboxes = []

    for _,row in image_bboxes.iterrows():

        bboxes.append((row.bbox_xmin, row.bbox_ymin, row.bbox_width, row.bbox_height))

        

    return bboxes



def plot_image_examples(df, rows=3, cols=3, title='Image examples'):

    fig, axs = plt.subplots(rows, cols, figsize=(50,50))

    for row in range(rows):

        for col in range(cols):

            idx = np.random.randint(len(df), size=1)[0]

            img_id = df.iloc[idx].image_id

            

            img = Image.open(TRAIN_DIR + img_id + '.jpg')

            axs[row, col].imshow(img)

            

            bboxes = get_all_bboxes(df, img_id)

            

            for bbox in bboxes:

                rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=3,edgecolor='r',facecolor='none')

                axs[row, col].add_patch(rect)

            

            axs[row, col].axis('off')

            

    plt.suptitle(title)
plot_image_examples(sorted_df[sorted_df["area"]<5000])
plot_image_examples(sorted_df[sorted_df["area"]<2000])


plot_image_examples(sorted_df[sorted_df["area"]<1000])
plot_image_examples(sorted_df[(sorted_df["area"]>200000) & (sorted_df["area"]<300000)])
filtered_df=sorted_df.drop(sorted_df[(sorted_df["area"]>200000) | (sorted_df["area"]<2000)].index)
filtered_df.reset_index(drop=True, inplace=True)
filtered_df[(filtered_df["bbox_height"]>1024) | (filtered_df["bbox_height"]>1024)]
filtered_df[(filtered_df["bbox_xmin"]>1024) | (filtered_df["bbox_ymin"]>1024)]
print(f"total bboxes before cleaning {len(train_data)}")

print(f"total bboxes after cleaning {len(filtered_df)}")

print(f"total bboxes cleaned {len(train_data)-len(filtered_df)}")
filtered_df.to_csv("cleaned_data.csv",index=False)