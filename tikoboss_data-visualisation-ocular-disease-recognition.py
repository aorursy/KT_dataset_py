import warnings

warnings.filterwarnings("ignore")

import os, glob, cv2

import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.layers import *

from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import get_custom_objects

#import efficientnet.tfkeras as efn

from tqdm import tqdm

sns.set_style("darkgrid")
images_paths = '../input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images/'

#train_df= pd.read_excel('../input/ocular-disease-recognition-odir5k/ODIR-5K/data.xlsx')
# code utiliser pour converter un fichier xlsx en csv

"""

import pandas as pd

data_xls = pd.read_excel('../input/ocular-disease-recognition-odir5k/ODIR-5K/data.xlsx', dtype=str, index_col=None)

data_xls.to_csv('ocular_disease_csv_file.csv', encoding='utf-8', index=False)

"""
BASEPATH = "../input/cvs-file"

train_df = pd.read_csv(os.path.join(BASEPATH, 'ocular_disease_csv_file.csv'))
#nombre des images dans notre data-set

import os

list = os.listdir(images_paths) 

number_files = len(list)

print (number_files)
print('Train data a {:,} images'.format(len(train_df)))
train_df.head()
train_df.columns = ["id", 'age', "sex", "left_fundus", "right_fundus", "left_diagnosys", "right_diagnosys", "normal",

                    "diabetes", "glaucoma", "cataract", "amd", "hypertension", "myopia", "other"]
train_df.head()
import seaborn as sns

fig, (ax1) = plt.subplots(1, 1, figsize=(20,5))

sns.countplot(ax=ax1, x="age", data=train_df)

ax1.set_title("distribution d'Age  dans  train_df")

plt.show()
import seaborn as sns

fig, (ax1) = plt.subplots(1, 1, figsize=(20,5))

sns.countplot(ax=ax1, x="sex", data=train_df)

ax1.set_title("distribution de Sex  dans  train_df")

plt.show()

sns.set_style("darkgrid")

fig= plt.subplots(figsize=(20,5))

sns.countplot(x='sex', hue="diabetes", data=train_df)

plt.title("Diagnostics normaux regroupés par sexe")

plt.show()

sns.set_style("darkgrid")

fig= plt.subplots(figsize=(20,5))

sns.countplot(x='sex', hue="diabetes", data=train_df)

plt.title("Diagnostics de diabète regroupés par sexe")

plt.show()

sns.set_style("darkgrid")

fig= plt.subplots(figsize=(20,5))

sns.countplot(x='sex', hue="glaucoma", data=train_df)

plt.title("Diagnostics de glaucoma regroupés par sexe")

plt.show()
sns.set_style("darkgrid")

fig= plt.subplots(figsize=(20,5))

sns.countplot(x='sex', hue="cataract", data=train_df)

plt.title("Diagnostics du cataract regroupés par sexe")

plt.show()
sns.set_style("darkgrid")

fig= plt.subplots(figsize=(20,5))

sns.countplot(x='sex', hue="amd", data=train_df)

plt.title("Diagnostics du amd regroupés par sexe")

plt.show()
sns.set_style("darkgrid")

fig= plt.subplots(figsize=(20,5))

sns.countplot(x='sex', hue="hypertension", data=train_df)

plt.title("Diagnostics de l'hypertension regroupés par sexe")

plt.show()
sns.set_style("darkgrid")

fig= plt.subplots(figsize=(20,5))

sns.countplot(x='sex', hue="myopia", data=train_df)

plt.title("Diagnostics de la myopie regroupés par sexe")

plt.show()
sns.set_style("darkgrid")

fig= plt.subplots(figsize=(20,5))

sns.countplot(x='sex', hue="other", data=train_df)

plt.title("Diagnostics des autres maladies regroupés par sexe")

plt.show()
def plot_count(feature, title, df, size=1, show_all=False):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    if show_all:

        g = sns.countplot(df[feature], palette='Set3')

        g.set_title("{} distribution".format(title))

    else:

        g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

        if(size > 2):

            plt.xticks(rotation=90, size=8)

            for p in ax.patches:

                height = p.get_height()

                ax.text(p.get_x()+p.get_width()/2.,

                        height + 0.2,

                        '{:1.2f}%'.format(100*height/total),

                        ha="center") 

        g.set_title("Nombre et pourcentage de {}".format(title))

    plt.show()   
plot_count("left_diagnosys", "Diagnostic de l'œil gauche", train_df, size=4)
plot_count("right_diagnosys", "Diagnostic de l'œil droite", train_df, size=4)
import imageio

IMAGE_PATH = "/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images"

def show_images(df, title="Diagnosys", eye_exam="left_fundus"):

    print(f"{title}; eye exam: {eye_exam}")

    f, ax = plt.subplots(3,3, figsize=(16,16))

    for i,idx in enumerate(df.index):

        dd = df.iloc[idx]

        image_name = dd[eye_exam]

        image_path = os.path.join(IMAGE_PATH, image_name)

        img_data=imageio.imread(image_path)

        ax[i//3, i%3].imshow(img_data)

        ax[i//3, i%3].axis('off')

    plt.show()

df = train_df.loc[(train_df.normal==1)].sample(9).reset_index()

show_images(df,title="Oeil gauche normal",eye_exam="left_fundus")
df = train_df.loc[(train_df.normal==1)].sample(9).reset_index()

show_images(df,title="Oeil droite normal",eye_exam="right_fundus")
df = train_df.loc[(train_df.cataract==1) & (train_df.left_diagnosys=="cataract")].sample(9).reset_index()

show_images(df,title="Oeil gauche avec cataracte",eye_exam="left_fundus")

df = train_df.loc[(train_df.cataract==1) & (train_df.right_diagnosys=="cataract")].sample(9).reset_index()

show_images(df,title="Oeil droite avec cataracte",eye_exam="right_fundus")
df = train_df.loc[(train_df.glaucoma==1) & (train_df.left_diagnosys=="glaucoma")].sample(9).reset_index()

show_images(df,title="œil gauche avec glaucome",eye_exam="left_fundus")
df = train_df.loc[(train_df.glaucoma==1) & (train_df.right_diagnosys=="glaucoma")].sample(9).reset_index()

show_images(df,title="œil droite avec glaucome",eye_exam="right_fundus")
df = train_df.loc[(train_df.myopia==1)].sample(9).reset_index()

show_images(df,title="Oeil gauche avec myopie",eye_exam="left_fundus") 
df = train_df.loc[(train_df.myopia==1)].sample(9).reset_index()

show_images(df,title="Oeil droite avec myopie",eye_exam="right_fundus")
df = train_df.loc[(train_df.amd==1)].sample(9).reset_index()

show_images(df,title="Oeil gauche avec AMD ",eye_exam="left_fundus")
df = train_df.loc[(train_df.amd==1)].sample(9).reset_index()

show_images(df,title="Oeil droite avec AMD ",eye_exam="right_fundus")
df = train_df.loc[(train_df.hypertension==1)].sample(9).reset_index()

show_images(df,title="Oeil gauche avec hypertension ",eye_exam="left_fundus")
df = train_df.loc[(train_df.hypertension==1)].sample(9).reset_index()

show_images(df,title="Oeil droite avec hypertension ",eye_exam="right_fundus")
df = train_df.loc[(train_df.other==1)].sample(9).reset_index()

show_images(df,title="Oeil gauche avec une autre maladie",eye_exam="left_fundus")
df = train_df.loc[(train_df.hypertension==1)].sample(9).reset_index()

show_images(df,title="Oeil droite avec une autre maladie ",eye_exam="right_fundus")