import os

import random
import seaborn as sns
import cv2

# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
import IPython.display as ipd
import glob
import h5py
import plotly.graph_objs as go
import plotly.express as px
from PIL import Image
from tempfile import mktemp

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, LinearAxis, Range1d
from bokeh.models.tools import HoverTool
from bokeh.palettes import BuGn4
from bokeh.plotting import figure, output_notebook, show
from bokeh.transform import cumsum
from math import pi

output_notebook()

from IPython.display import Image, display
import warnings
warnings.filterwarnings("ignore")
os.listdir('../input/landmark-recognition-2020/')
BASE_PATH = '../input/landmark-recognition-2020'

TRAIN_DIR = f'{BASE_PATH}/train'
TEST_DIR = f'{BASE_PATH}/test'

print('Reading data...')
train = pd.read_csv(f'{BASE_PATH}/train.csv')
submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
print('Reading data completed')
display(train.head())
print("Shape of train_data :", train.shape)
display(submission.head())
print("Shape of submission :", submission.shape)
# displaying only top 30 landmark
landmark = train.landmark_id.value_counts()
landmark_df = pd.DataFrame({'landmark_id':landmark.index, 'frequency':landmark.values}).head(30)

landmark_df['landmark_id'] =   landmark_df.landmark_id.apply(lambda x: f'landmark_id_{x}')

fig = px.bar(landmark_df, x="frequency", y="landmark_id",color='landmark_id', orientation='h',
             hover_data=["landmark_id", "frequency"],
             height=1000,
             title='Number of images per landmark_id (Top 30 landmark_ids)')
fig.show()
import PIL
from PIL import Image, ImageDraw


def display_images(images, title=None): 
    f, ax = plt.subplots(5,5, figsize=(18,22))
    if title:
        f.suptitle(title, fontsize = 30)

    for i, image_id in enumerate(images):
        image_path = os.path.join(TRAIN_DIR, f'{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg')
        image = Image.open(image_path)
        
        ax[i//5, i%5].imshow(image) 
        image.close()       
        ax[i//5, i%5].axis('off')

        landmark_id = train[train.id==image_id.split('.')[0]].landmark_id.values[0]
        ax[i//5, i%5].set_title(f"ID: {image_id.split('.')[0]}\nLandmark_id: {landmark_id}", fontsize="12")

    plt.show() 
samples = train.sample(25).id.values
display_images(samples)
samples = train[train.landmark_id == 138982].sample(25).id.values

display_images(samples)
samples = train[train.landmark_id == 126637].sample(25).id.values

display_images(samples)
samples = train[train.landmark_id == 20409].sample(25).id.values

display_images(samples)
samples = train[train.landmark_id == 83144].sample(25).id.values

display_images(samples)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') ## ???
import plotly_express as px
import plotly.graph_objects as go
import glob
from tqdm.notebook import tqdm
import cv2
import os
import random
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('/kaggle/input/landmark-recognition-2020/train.csv')
test = glob.glob('/kaggle/input/landmark-recognition-2020/test/*/*/*/*.jpg')
print('Total Train Images: {}'.format(len(df_train))) 
print('Total Test Images: {}'.format(len(test)))
print('Total Unique Landmarks: {}'.format(df_train.landmark_id.nunique()))
landmarks = df_train.groupby('landmark_id',as_index=False)['id'].count()\
    .sort_values('id',ascending=False).reset_index(drop=True)
landmarks.rename(columns={'id':'count'},inplace=True)
def add_text(ax,fontsize=12):
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{}'.format(int(y)), (x.mean(), y), ha='center', va='bottom',size=fontsize)
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(16,8))
sns.barplot(data=landmarks[:50],x='landmark_id',y='count',ax=ax1,color='#30a2da',
           order=landmarks[:50]['landmark_id'])
add_text(ax1,fontsize=8)
ax1.set_title('Top 50 Landmarks')
ax1.set_ylabel('Number of Images')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right",size=8)
sns.barplot(data=landmarks[-50:],x='landmark_id',y='count',ax=ax2,color='#fc4f30')
ax2.set_title('Bottom 50 Landmarks')
ax2.set_ylabel('Number of Images')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right",size=8)
plt.tight_layout()
print(f"Number of Landmarks with less than 10 images are {len(landmarks[landmarks['count']<10])}")
print(f"Number of Landmarks with less than 20 images are {len(landmarks[landmarks['count']<20])}")
plt.show()
plt.figure(figsize=(16,4))
ax = sns.distplot(df_train['landmark_id'],bins=500)
ax.set_title('Distribution of Landmarks')
plt.tight_layout()
plt.show()
def get_image(id):
    path = os.path.join('/kaggle/input/landmark-recognition-2020/train',
                        id[0],id[1],id[2],id+'.jpg')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def show_data(df,rows,cols):
    df.reset_index(inplace=True,drop=True)
    fig = plt.figure(figsize=(24,24))
    i = 1
    for r in range(rows):
        for c in range(cols):
            id = df.loc[i-1,'id']
            label = df.loc[i-1,'landmark_id']
            ax = fig.add_subplot(rows,cols,i)
            img = get_image(id)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label)
            ax.imshow(img)
            i+=1
    return fig
# 랜덤하게 가져온다
inds = np.random.choice(df_train.index.tolist(),20)
fig = show_data(df_train.iloc[inds,:],4,5)
fig.tight_layout()
df_images = df_train.drop_duplicates(subset=['landmark_id'])
df_images = df_images.sample(n=1000,random_state=23)
df_images.reset_index(inplace=True,drop=True)
df_images['height'] = 0
df_images['width'] = 0
df_images['channels'] = 0
for i in tqdm(range(len(df_images))):
    img = get_image(df_images.loc[i,'id'])
    df_images.loc[i,'height'] = img.shape[0]
    df_images.loc[i,'width'] = img.shape[1]
    df_images.loc[i,'channels'] = img.shape[2]
def img_distribution(df):
    shape = (np.min(df['width']), np.max(df['width']),
            np.min(df['height']), np.max(df['height']))
    fig = px.scatter(df,x='width',y='height')
    fig.add_shape(
        x0 = shape[0],
        x1 = shape[1],
        y0 = shape[2],
        y1 = shape[3],
        fillcolor = 'yellow',
        opacity=0.3,
        layer='below'
    )
    fig.add_trace(go.Scatter(name='mean',x=[np.mean(df['width'])],y=[np.mean(df['height'])],
                         marker=dict(color='red',size=10)))
    #fig.update_traces(marker_line_color='black',marker_line_width=1)
    fig.update_layout(width=700,height=400,margin=dict(l=0,b=0,r=0,t=40),template='seaborn',
                 title='Distribution of Image Dimensions', showlegend=False,
                 xaxis=dict(title='Width', mirror=True, linewidth=2, linecolor='black',showgrid=False),
                 yaxis=dict(title='Height', mirror=True, linewidth=2, linecolor='black',showgrid=False),
                 plot_bgcolor='rgb(255,255,255)')
    return fig
img_distribution(df_images)
