import pandas as pd

import numpy as np

import os

from PIL import Image, ImageDraw

from ast import literal_eval

import matplotlib.pyplot as plt



import plotly.express as px

import plotly.graph_objects as go



!ls ../input/global-wheat-detection/
root_path = "../input/global-wheat-detection/"

train_folder = os.path.join(root_path, "train")

test_folder = os.path.join(root_path, "test")

train_csv_path = os.path.join(root_path, "train.csv")

sample_submission = os.path.join(root_path, "sample_submission.csv")
train_df = pd.read_csv(train_csv_path)
train_df.head()
import ast

train_df['bbox_area'] = train_df.bbox.apply(lambda x: ast.literal_eval(x)[2] * ast.literal_eval(x)[3])
source_vc = train_df.source.value_counts()



print(f"There are {len(source_vc)} sources.\n")

print(source_vc)

summary_df = train_df.groupby(['source']).agg(unq_images=('image_id','nunique'), wheat_heads=('image_id','size'), mean_bbox_area=('bbox_area','mean'), median_bbox_area=('bbox_area','median'))

summary_df.reset_index(inplace=True,drop=False)

summary_df['mean_bbox_area'] = np.round(summary_df.mean_bbox_area, 2)

summary_df['median_bbox_area'] = np.round(summary_df.median_bbox_area, 2)

summary_df['mean_wheat_heads_per_img'] = np.round(summary_df.wheat_heads / summary_df.unq_images, 2)



summary_df

summary_df = train_df.groupby(['image_id']).agg(unq_source=('source','nunique'))

summary_df.reset_index(inplace=True,drop=False)

print(f"Images with >1 sources : {summary_df.loc[summary_df.unq_source > 1].shape[0]}")
print(f"Unique Widths: {pd.unique(train_df.width)}")

print(f"Unique HeightsA: {pd.unique(train_df.height)}")



summary_df = train_df.groupby(['source','image_id']).agg(wheat_heads=('image_id','size'), mean_bbox_area=('bbox_area','mean'))

summary_df.reset_index(inplace=True,drop=False)

summary_df









colors = ['rgba(93, 164, 214, 1.0)', 'rgba(255, 144, 14, 1.0)', 'rgba(44, 160, 101, 1.0)',

          'rgba(255, 65, 54, 1.0)', 'rgba(207, 114, 255, 1.0)', 'rgba(127, 96, 0, 1.0)', 'rgba(200, 200, 0, 1.0)']



fig = go.Figure()

index = 0

for cls in pd.unique(summary_df.source):

    fig.add_trace(go.Box(y=summary_df.loc[summary_df.source==cls,'wheat_heads'], name=cls,

                marker_color = colors[index]))

    index = index + 1



fig.update_yaxes(range=[0, 100])

fig.update_layout(title='Box-Plot Wheatheads w.r.t source',xaxis_title="Source",yaxis_title="No. Wheatheads per image")

fig.show()







fig = go.Figure()

index = 0

for cls in pd.unique(summary_df.source):

    fig.add_trace(go.Box(y=summary_df.loc[summary_df.source==cls,'mean_bbox_area'], name=cls,

                marker_color = colors[index]))

    index = index + 1



fig.update_yaxes(range=[0, 40000])

fig.update_layout(title='Box-Plot BBox Area w.r.t source',xaxis_title="Source",yaxis_title="Mean BBox Area")

fig.show()

def show_images(images, num = 5):

    

    images_to_show = np.random.choice(images, num)



    for image_id in images_to_show:



        image_path = os.path.join(train_folder, image_id + ".jpg")

        image = Image.open(image_path)



        # get all bboxes for given image in [xmin, ymin, width, height]

        bboxes = [literal_eval(box) for box in train_df[train_df['image_id'] == image_id]['bbox']]



        # visualize them

        draw = ImageDraw.Draw(image)

        for bbox in bboxes:    

            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=3)



        plt.figure(figsize = (15,15))

        plt.imshow(image)

        plt.show()
for source in pd.unique(train_df.source):

    print(f"Showing images for {source}:")

    show_images(train_df[train_df['source'] == source]['image_id'].unique(), num = 1)
