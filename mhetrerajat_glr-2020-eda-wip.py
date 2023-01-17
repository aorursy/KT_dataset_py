# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pathlib import Path
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import urllib.parse
np.random.seed(25)
BASE_DIR = Path("/kaggle/input/landmark-retrieval-2020")
TRAIN_CSV = BASE_DIR / "train.csv"
TRAIN_DATA = BASE_DIR / "train"
TEST_DATA = BASE_DIR / "test"
IDX_DATA = BASE_DIR / "index"
df = pd.read_csv(TRAIN_CSV)
df.shape
df.head(5)
# check if any data is missing
df.isna().sum()
# Get the number of images in each folders

train_imgs = TRAIN_DATA.rglob("*.jpg")
reduce(lambda acc, e: acc + 1, train_imgs, 0)
idx_imgs = IDX_DATA.rglob("*.jpg")
reduce(lambda acc, e: acc + 1, idx_imgs, 0)
test_imgs = TEST_DATA.rglob("*.jpg")
reduce(lambda acc, e: acc + 1, test_imgs, 0)
# Check if id i.e image id column is unique, as mentioned in the data description
df['id'].is_unique
# Get the number of images per landmark
landmark_dist = df['landmark_id'].value_counts().rename_axis('landmark_id')\
.reset_index(name="count")\
.sort_values(by=['count'], ascending=[False])

("Max : {} | Min : {}".format(landmark_dist['count'].max(), landmark_dist['count'].min()))
landmark_dist['count'].describe()
# Check the distribution of landmark's images
sns.set(style="darkgrid")
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(20, 12)
sns.countplot(
    df['landmark_id'],
    order=df['landmark_id'].value_counts().index[:25],
    ax=ax1
)
sns.countplot(
    df['landmark_id'],
    order=df['landmark_id'].value_counts().index[-25:],
    ax=ax2
)
ax1.set_title('distribution of landmark images | top 25')
ax2.set_title('distribution of landmark images | last 25')
ax2.set(ylim=(0, 50))
plt.show()
# Courtesy :: https://www.kaggle.com/sudeepshouche/identify-landmark-name-from-landmark-id

url = 'https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv'
df_classes = pd.read_csv(url, index_col = 'landmark_id', encoding='latin', engine='python')

get_landmark_name = lambda x: urllib.parse.unquote(x['category'].replace('http://commons.wikimedia.org/wiki/Category:', ''))
df_classes['landmark_name'] = df_classes.apply(get_landmark_name, axis=1)
df_classes.head(5)
# Randomly chooses at max 12 images for any landmark based on its id
def render(img_path, nrow, col, ax, row):
    print("Loading : {}".format(img_path))
    img = cv2.imread(img_path)
    ax[nrow, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if row is not None:
        ax[nrow, col].set_title(row['landmark_name'])


def get_images(landmark_id):    
    _df = df.loc[df['landmark_id'] == landmark_id, :]
    
    plt.rcParams["axes.grid"] = False
    
    _df = _df.sample(n=12).reset_index()
    
    _df = pd.merge(_df, df_classes, on=['landmark_id'], how="inner")
    
    no_row = np.math.ceil(min(len(_df), 12)/3)
    f, ax = plt.subplots(no_row, 3, figsize=(24, 20))
    print("Fig Shape : {0}".format((no_row, 3)))
        
    
    nrow = 0
    for idx, row in _df.iterrows():
        image_id = row['id'] + ".jpg"
        
        img_path = TRAIN_DATA / image_id[0] / image_id[1] / image_id[2] / image_id
        
        col = int(idx % 3)
        render(str(img_path), nrow, col, ax, row)
                
        # when all columns are filled in a row
        if col == 2:
            nrow += 1
    
get_images(138982)
# 2nd Landmark Imgs
get_images(126637)
# Get random images from folders

def get_images_from_dir(img_dir):
    imgs = np.random.choice(list(TEST_DATA.rglob("*.jpg")), 12)
    
    fig, ax = plt.subplots(4, 3, figsize=(24, 20))
    
    row = 0
    for idx, img_path in enumerate(imgs):
        
        col = idx % 3
        
        render(str(img_path), row, col, ax, row=None)
        
        if col == 2:
            row += 1
# Random images from test dir
get_images_from_dir(TEST_DATA)
# Random images from index dir
get_images_from_dir(IDX_DATA)
