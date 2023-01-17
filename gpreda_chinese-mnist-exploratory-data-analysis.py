import pandas as pd

import numpy as np

import sys

import os

import cv2 as cv

import matplotlib.pyplot as plt

import seaborn as sns

import skimage

import skimage.io
IMAGE_PATH = '..//input//chinese-mnist//data//data//'

IMAGE_WIDTH = 64

IMAGE_HEIGHT = 64

IMAGE_CHANNELS = 1

RANDOM_STATE = 42
os.listdir("..//input//chinese-mnist")
data_df=pd.read_csv('..//input//chinese-mnist//chinese_mnist.csv')
data_df.shape
data_df.sample(100).head()
def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data(data_df)
image_files = list(os.listdir(IMAGE_PATH))

print("Number of image files: {}".format(len(image_files)))
def create_file_name(x):

    file_name = f"input_{x[0]}_{x[1]}_{x[2]}.jpg"

    return file_name
data_df["file"] = data_df.apply(create_file_name, axis=1)
data_df.head()
file_names = list(data_df['file'])

print("Matching image names: {}".format(len(set(file_names).intersection(image_files))))
def read_image_sizes(file_name):

    image = skimage.io.imread(IMAGE_PATH + file_name)

    return list(image.shape)
m = np.stack(data_df['file'].apply(read_image_sizes))

df = pd.DataFrame(m,columns=['w','h'])

data_df = pd.concat([data_df,df],axis=1, sort=False)
print(f"Images widths #: {data_df.w.nunique()},  heights #: {data_df.h.nunique()}")

print(f"Images widths values: {data_df.w.unique()},  heights values: {data_df.h.unique()}")
data_df.head()
print(f"Number of suites: {data_df.suite_id.nunique()}")

print(f"Samples: {data_df.sample_id.nunique()}: {list(data_df.sample_id.unique())}")

print(f"Characters codes: {data_df.code.nunique()}: {list(data_df.code.unique())}")

print(f"Characters: {data_df.character.nunique()}: {list(data_df.character.unique())}")

print(f"Numbers: {data_df.value.nunique()}: {list(data_df.value.unique())}")
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set2')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()  
plot_count("code", "character code", data_df, size=3)
plot_count("value", "number value", data_df, size=3)
print(f"frequence of each character:")

data_df.character.value_counts()
def show_images(df, isTest=False):

    f, ax = plt.subplots(10,15, figsize=(15,10))

    for i,idx in enumerate(df.index):

        dd = df.iloc[idx]

        image_name = dd['file']

        image_path = os.path.join(IMAGE_PATH, image_name)

        img_data = cv.imread(image_path)

        ax[i//15, i%15].imshow(img_data)

        ax[i//15, i%15].axis('off')

    plt.show()
df = data_df.loc[data_df.suite_id==1].sort_values(by=["sample_id","value"]).reset_index()

show_images(df)
df = data_df.loc[data_df.suite_id==37].sort_values(by=["sample_id","value"]).reset_index()

show_images(df)
df = data_df.loc[data_df.suite_id==75].sort_values(by=["sample_id","value"]).reset_index()

show_images(df)
df = data_df.loc[data_df.code==1].sample(150).reset_index()

show_images(df)
df = data_df.loc[data_df.code==5].sample(150).reset_index()

show_images(df)