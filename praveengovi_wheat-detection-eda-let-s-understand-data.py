from IPython.display import HTML

HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/NlpS-DhayQA?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import cv2

import math

import os, ast

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt # plotting

import matplotlib.patches as patches



import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px

import plotly.figure_factory as ff

from sklearn.preprocessing import OneHotEncoder



import seaborn as sns

from tqdm import tqdm

import matplotlib.cm as cm

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.utils import shuffle



tqdm.pandas()

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

# Some constants

dataset_path = '/kaggle/input/global-wheat-detection'

dataset_img_train='/kaggle/input/global-wheat-detection/train/'

dataset_img_test='/kaggle/input/global-wheat-detection/test/'
%%time

train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))

sample_sub_df = pd.read_csv(os.path.join(dataset_path, 'sample_submission.csv'))
train_df.head()
sample_sub_df.head()
print(f'Shape of training data: {train_df.shape}')

print(f'Shape of given test data: {sample_sub_df.shape}')
%%time

SAMPLE_LEN=2000

def load_image(image_id):

    file_path = image_id + ".jpg"

    image = cv2.imread(dataset_img_train + file_path)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



train_images = train_df["image_id"][:SAMPLE_LEN].progress_apply(load_image)
fig = px.imshow(cv2.resize(train_images[99], (205, 136)))

fig.show()
print(f'Total no. of images                        : {train_df.shape[0]}')

print(f'Total no. of unique images                 : {train_df["image_id"].nunique()}')

print(f'Checking Dimentions - heights and widths   : {train_df["width"].unique()}, {train_df["height"].unique()}')

print(f'Maximum number of wheat heads in the Image : {max(train_df["image_id"].value_counts())}')

print(f'Average wheat heads in the Image           : {len(train_df)/train_df["image_id"].nunique()}')
sns.distplot(train_df['image_id'].value_counts(), kde=True)

plt.xlabel('# of wheat heads')

plt.ylabel('# of images')

plt.title('# of wheat heads vs. # of images')

plt.show()
box_count = train_df["image_id"].value_counts()



hist_data = [box_count.values]

group_labels = ['Count'] # name of the dataset



fig = ff.create_distplot(hist_data, group_labels, bin_size=2)

fig.update_layout(title_text="Number of bounding boxes per image", template="simple_white", title_x=0.5)

fig.show()
train_df[['x_min','y_min', 'width', 'height']] = pd.DataFrame([ast.literal_eval(x) for x in train_df.bbox.tolist()], index= train_df.index)

train_df = train_df[['image_id', 'bbox', 'source', 'x_min', 'y_min', 'width', 'height']]

train_df
# Visualize few samples of current training dataset

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

count=1000

for row in ax:

    for col in row:

        img = plt.imread(f'{os.path.join(dataset_path, "train", train_df["image_id"].unique()[count])}.jpg')

        col.grid(False)

        col.set_xticks([])

        col.set_yticks([])

        col.imshow(img)

        count += 1

plt.show()
##  Thanks to https://www.kaggle.com/kaushal2896/global-wheat-detection-starter-eda kernal , Kindly upvote this kernal also



def get_bbox(image_id, df, col, color='white'):

    bboxes = df[df['image_id'] == image_id]

    

    for i in range(len(bboxes)):

        # Create a Rectangle patch

        rect = patches.Rectangle(

            (bboxes['x_min'].iloc[i], bboxes['y_min'].iloc[i]),

            bboxes['width'].iloc[i], 

            bboxes['height'].iloc[i], 

            linewidth=2, 

            edgecolor=color, 

            facecolor='none')



        # Add the patch to the Axes

        col.add_patch(rect)
# Visualize few samples of current training dataset

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

count=0

for row in ax:

    for col in row:

        img_id = train_df["image_id"].unique()[count]

        img = plt.imread(f'{os.path.join(dataset_path, "train", img_id)}.jpg')

        col.grid(False)

        col.set_xticks([])

        col.set_yticks([])

        get_bbox(img_id, train_df, col, color='red')

        col.imshow(img)

        count += 1

plt.show()
image_id = (train_df['image_id'].value_counts() == max(train_df["image_id"].value_counts())).index[0]

print('Maximum wheat heads :',max(train_df["image_id"].value_counts()))

img = plt.imread(f'{os.path.join(dataset_path, "train", image_id)}.jpg')

fig, ax = plt.subplots(1, figsize=(12, 12))

ax.grid(False)

ax.set_xticks([])

ax.set_yticks([])

ax.axis('off')

get_bbox(image_id, train_df, ax, color='orange')

ax.imshow(img)

plt.plot()
source = train_df['source'].value_counts()

print(train_df['source'].value_counts())

wheat_src_df = train_df.groupby(['source']).agg({'image_id':'count'}).reset_index()

wheat_src_df.rename(columns={'image_id':'count'},inplace=True)
wheat_src_df
fig = go.Figure(data=[

    go.Pie(labels=source.index, values=source.values)

])



fig.update_layout(title='Source distribution')

fig.show()
fig = go.Figure(go.Bar(x=train_df['source'].value_counts().index, 

                       y=train_df['source'].value_counts(),

                       marker_color='lightsalmon'))

fig.update_layout(title_text="Bar chart of sources", title_x=0.5)

fig.show()
bbox_count = train_df.groupby("source")["image_id"].apply(lambda X: X.value_counts().mean()).reset_index().rename(columns={"image_id": "bbox_count"})



fig = go.Figure(go.Bar(x=bbox_count.source, 

                       y=bbox_count.bbox_count,

                       name='Bbox counts', marker_color='indianred'))

fig.update_layout(title_text="Bar chart of Bbox counts in image", template="simple_white", title_x=0.5)

fig.show()
red_values = [np.mean(train_images[idx][:, :, 0]) for idx in range(len(train_images))]

green_values = [np.mean(train_images[idx][:, :, 1]) for idx in range(len(train_images))]

blue_values = [np.mean(train_images[idx][:, :, 2]) for idx in range(len(train_images))]

values = [np.mean(train_images[idx]) for idx in range(len(train_images))]
fig = ff.create_distplot([values], group_labels=["Channels"], colors=["purple"])

fig.update_layout(showlegend=False, template="simple_white")

fig.update_layout(title_text="Distribution of channel values")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig
fig = ff.create_distplot([red_values], group_labels=["R"], colors=["red"])

fig.update_layout(showlegend=False, template="simple_white")

fig.update_layout(title_text="Distribution of red channel values")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig
fig = ff.create_distplot([blue_values], group_labels=["B"], colors=["blue"])

fig.update_layout(showlegend=False, template="simple_white")

fig.update_layout(title_text="Distribution of blue channel values")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig
fig = ff.create_distplot([green_values], group_labels=["G"], colors=["green"])

fig.update_layout(showlegend=False, template="simple_white")

fig.update_layout(title_text="Distribution of green channel values")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig
fig = go.Figure()



for idx, values in enumerate([red_values, green_values, blue_values]):

    if idx == 0:

        color = "Red"

    if idx == 1:

        color = "Green"

    if idx == 2:

        color = "Blue"

    fig.add_trace(go.Box(x=[color]*len(values), y=values, name=color, marker=dict(color=color.lower())))

    

fig.update_layout(yaxis_title="Mean value", xaxis_title="Color channel",

                  title="Mean value vs. Color channel", template="plotly_white")
fig = ff.create_distplot([red_values, green_values, blue_values],

                         group_labels=["R", "G", "B"],

                         colors=["red", "green", "blue"])

fig.update_layout(title_text="Distribution of Red,Blue,Green channel values", template="simple_white")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig.data[1].marker.line.color = 'rgb(0, 0, 0)'

fig.data[1].marker.line.width = 0.5

fig.data[2].marker.line.color = 'rgb(0, 0, 0)'

fig.data[2].marker.line.width = 0.5

fig
df_img_wht_heads = train_df.groupby(['image_id']).agg({'source':'count'}).reset_index().rename(columns={'source':'wheat_head_cnt'})
df_img_wht_heads.head()
df_img_wht_heads['wheat_head_cnt'].describe(include="all")
sns.boxplot(df_img_wht_heads['wheat_head_cnt'])
def catagory(col):

    if col >= 0 and col <= 28 :

        ctg="Less_Wheat_heads"

    elif col <= 43 and col >= 28:

        ctg="Medium_Wheat_heads"

    elif col <= 59 and col >= 43:

        ctg="High_Wheat_heads"

    else:

        ctg="Extra_High_Wheat_heads"

    return ctg
def binary_catagory(col):

    if col >= 0 and col <= 43 :

        ctg=0

    else:

        ctg=1

    return ctg
df_img_wht_heads['Wheat_head_catagory']=df_img_wht_heads['wheat_head_cnt'].apply(catagory)

df_img_wht_heads['Wheat_heads_ctg_High_Low']=df_img_wht_heads['wheat_head_cnt'].apply(binary_catagory)
fig = go.Figure(go.Bar(x=df_img_wht_heads['Wheat_head_catagory'].value_counts().index, 

                       y=df_img_wht_heads['Wheat_head_catagory'].value_counts(),

                       marker_color='lightsalmon'))

fig.update_layout(title_text="Bar chart of Wheat Head Catagory", title_x=0.5)

fig.show()
df_img_wht_heads['Wheat_head_catagory'].unique()
df_img_wht_heads.head()
# generate binary values using get_dummies

df_img_wht_heads = pd.get_dummies(df_img_wht_heads, columns=["Wheat_head_catagory"],prefix="")
df_img_wht_heads.columns
fig = px.parallel_categories(df_img_wht_heads[['_Extra_High_Wheat_heads', '_High_Wheat_heads', '_Less_Wheat_heads','_Medium_Wheat_heads']], \

                             color="_Less_Wheat_heads", color_continuous_scale="sunset",\

                             title="Parallel categories plot of targets")

fig