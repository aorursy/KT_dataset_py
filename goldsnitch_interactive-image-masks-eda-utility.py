import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib.patches as patches

from ipywidgets import interact, IntSlider, fixed

from pathlib import Path

from PIL import Image
base_path = Path('../input/global-wheat-detection/')

TRAIN_CSV_PATH = base_path / 'train.csv'

TRAIN_DIR = base_path / 'train/'
def get_all_bboxes(df, img_id):

    image_bboxes = df[df.image_id == img_id]      

    return image_bboxes[["x", "y", "w", "h"]]



def plot_image(df, img_id, train_dir=TRAIN_DIR): # using constant, be careful

    plt.figure(figsize=(9, 9))

    ax = plt.gca()

    

    img = Image.open(str(train_dir / (img_id + '.jpg')))

    ax.imshow(img)

    

    #plot bboxes on image

    bboxes = get_all_bboxes(df, img_id)

    for bbox in bboxes.iterrows():

        bbox = bbox[1] # 0 element is index

        coords = ((bbox.x, bbox.y), bbox.w, bbox.h)

        rect = patches.Rectangle(*coords, linewidth=1, edgecolor='r',facecolor='none')

        ax.add_patch(rect)



    plt.axis('off')
def interactive_plot(df):

    uniq = df["image_id"].unique()

    @interact(df=fixed(df), id_num=IntSlider(max=len(uniq)-1))

    def choose_show(df, id_num):

        img_id = uniq[id_num]

        plot_image(df, img_id)
def bbox2separate_cols(df):

    """

    If df contains bbox column of 'O' dtype, function creates columns "x", "y", "w", "h". 

    Otherwise throws ValueError.

    """

    try:

        bbox_items = df["bbox"].str.split(',', expand=True)

    except KeyError:

        raise ValueError("df has no bbox col")

        

    # convert splitted strings to numbers

    bbox_items[0] = bbox_items[0].transform(lambda x: x[1:]) #delete '[' and ']' symbols

    bbox_items[3] = bbox_items[3].transform(lambda x: x[:-1])

    bbox_items = bbox_items.astype(float)

    

    train[['x', 'y', 'w', 'h']] = bbox_items

    train.drop(columns=['bbox'], inplace=True)

    

    return train
train = pd.read_csv(TRAIN_CSV_PATH)

train = bbox2separate_cols(train)
interactive_plot(train)