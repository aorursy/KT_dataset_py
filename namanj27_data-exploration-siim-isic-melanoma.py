# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from os import listdir

import matplotlib.pyplot as plt

%matplotlib inline  

# To store resultimg plots/graphs in the notebook document below the respective code cells



!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

#Required to apply plotly

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



import seaborn as sns

sns.set(style='whitegrid')



import pydicom



import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')

plt.show()

orange_black = [

    '#fdc029', '#df861d', '#FF6347', '#aa3d01', '#a30e15', '#800000', '#171820'

]
print(os.listdir('../input/siim-isic-melanoma-classification/'))
IMAGE_PATH = '../input/siim-isic-melanoma-classification/'



train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')



#Training data

print('Number of Training examples', train_df.shape[0])

print('Number of Test examples', test_df.shape[0])

train_df.head()
z = train_df.groupby(['benign_malignant'])['target'].count().to_frame()

z.style.background_gradient(cmap='Oranges')

#missing values



print('----train_df-----')

print(train_df.info());



print('\n')

print('----test_df-----')

print(test_df.info());
print('Total images in training set ',train_df['image_name'].count())

print('Total images in test set ',test_df['image_name'].count())
print(f"There are total {train_df['patient_id'].count()} patient ids, out of which {train_df['patient_id'].value_counts().count()} are unique")
columns = train_df.columns.tolist()

columns
z = train_df['target'].value_counts().to_frame()

z.style.background_gradient(cmap='Oranges')
#interactive plots

train_df['target'].value_counts(normalize=True).iplot(kind='bar',

                                                     yTitle='Percentage',

                                                     linecolor='black',

                                                     opacity=0.7,

                                                     color='orange',

                                                     bargap=0.8,

                                                     gridcolor='white',

                                                     title='[INTERACTIVE] Target value distribution from training set')
z = train_df['sex'].value_counts().to_frame()

z.style.background_gradient(cmap='Oranges')
#interactive plots

train_df['sex'].value_counts(normalize=True).iplot(kind='bar',

                                                     yTitle='Percentage',

                                                     linecolor='black',

                                                     opacity=0.7,

                                                     color='purple',

                                                     bargap=0.8,

                                                     gridcolor='white',

                                                     title='[INTERACTIVE] Sex column distribution from training set')
z = train_df.groupby(['target','sex'])['benign_malignant'].count().to_frame().reset_index()

z.style.background_gradient(cmap='Oranges')

sns.catplot(x='target', y='benign_malignant',hue='sex', data=z, kind='bar',palette=orange_black);

plt.xlabel('Benign:0   Malignant:1')

plt.ylabel('Count')
z = train_df['anatom_site_general_challenge'].value_counts(normalize=True).sort_values(ascending=True).to_frame()

z.style.background_gradient(cmap='Oranges')
train_df['anatom_site_general_challenge'].value_counts(normalize=True).sort_values().iplot(kind='barh',

                                                                                          xTitle='Percentage',

                                                                                          linecolor='black',

                                                                                          opacity=0.7,

                                                                                          color='orange',

                                                                                          theme='pearl',

                                                                                          bargap=0.2,

                                                                                          gridcolor='white',

                                                                                          title='[INTERACTIVE] Distribution of imaged site of intrerest from training set')
z = train_df.groupby(['sex', 'anatom_site_general_challenge'])['benign_malignant'].count().to_frame().reset_index()

z.style.background_gradient(cmap='Oranges')
sns.catplot(x='anatom_site_general_challenge', y='benign_malignant', hue='sex', data=z, kind='bar', palette=orange_black)

plt.gcf().set_size_inches(12,8)

plt.xlabel('Location of imaged site')

plt.xticks(rotation=45, fontsize=15)

plt.ylabel('# of Melanoma cases')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,8))



sns.distplot(train_df.age_approx,ax=axes[0], label='Train', color='#fdc029')

sns.distplot(test_df.age_approx,ax=axes[0], label='Test', color='#171820')

axes[0].set_title('Age distribution in train/test sets')

axes[0].legend()





sns.distplot(train_df[train_df.sex=='female'].age_approx,ax=axes[1], label='Female', color='#fdc029')

sns.distplot(train_df[train_df.sex=='male'].age_approx,ax=axes[1], label='Female', color='#171820')

axes[1].set_title('Age distribution w.r.t gender')

axes[1].legend()





plt.tight_layout()

plt.show()
z = train_df['diagnosis'].value_counts().sort_values().to_frame()

z.style.background_gradient(cmap='Oranges')
train_df['diagnosis'].value_counts(normalize=True).sort_values().iplot(kind='barh',

                                                          xTitle='Percentage',

                                                          linecolor='black',

                                                          opacity=0.7,

                                                          color='orange',

                                                          theme='pearl',

                                                          bargap=0.2,

                                                          gridcolor='white',

                                                          title='[INTERACTIVE] Distribution of Diagnosis column from training set')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

# kdeplot of age_approx for benign target

sns.kdeplot(train_df.loc[train_df['target'] == 0, 'age_approx'],ax=axes[0], label='Benign',color='g', shade=True)



# kdeplot of age_approx for Malignant target

sns.kdeplot(train_df.loc[train_df['target'] == 1, 'age_approx'],ax=axes[0], label='Malignant',color='b', shade=True)



axes[0].set_xlabel('Age in years')

axes[0].set_ylabel('Density')

axes[0].set_title('Age Distribution [Benign/Malignant]')



# kdeplot of age_approx for male gender

sns.kdeplot(train_df.loc[train_df['sex'] == 'male', 'age_approx'],ax=axes[1], label='male',color='g', shade=True)



# kdeplot of age_approx for female gender

sns.kdeplot(train_df.loc[train_df['sex'] == 'female', 'age_approx'],ax=axes[1], label='female',color='b', shade=True)



axes[1].set_xlabel('Gender')

axes[1].set_ylabel('Density')

axes[1].set_title('Age Distribution [Male/Female]')
# Location of the image dir

img_dir = IMAGE_PATH+'jpeg/train'
# Benign

benign = train_df[train_df['benign_malignant']=='benign']



f = plt.figure(figsize=(16,8))

f.add_subplot(1,2,1)



sample_img = benign['image_name'][0]+'.jpg'

image = plt.imread(os.path.join(img_dir, sample_img))

plt.imshow(image, cmap='gray')

plt.colorbar()

plt.title('Benign Image')

print(f"Image dimensions {image.shape}")

print(f"Maximum pixel value {image.max():.2f}; Minimum pixel value {image.min():.2f}")

print(f"Mean value of the pixels : {image.mean():.2f} ; Standard deviation : {image.std():.2f}")



f.add_subplot(1,2,2)



_ = plt.hist(image[:,:,0].ravel(), bins = 256, color = 'red', alpha = 0.5)

_ = plt.hist(image[:,:,1].ravel(), bins = 256, color = 'green', alpha = 0.5)

_ = plt.hist(image[:,:,2].ravel(), bins = 256, color = 'blue', alpha = 0.5)

_ = plt.xlabel('Intensity Values')

_ = plt.ylabel('Count')

_ = plt.legend(['red_channel','green_channel','blue_channel'])

plt.show()



#Malignant

malignant = train_df[train_df['benign_malignant']=='malignant']



f = plt.figure(figsize=(16,8))

f.add_subplot(1,2,1)



sample_img = malignant['image_name'][91]+'.jpg'

image = plt.imread(os.path.join(img_dir, sample_img))

plt.imshow(image, cmap='gray')

plt.colorbar()

plt.title('Malignant Image')

print(f"Image dimensions {image.shape}")

print(f"Maximum pixel value {image.max():.2f}; Minimum pixel value {image.min():.2f}")

print(f"Mean value of the pixels : {image.mean():.2f}; Standard deviation : {image.std():.2f}")



f.add_subplot(1,2,2)



_ = plt.hist(image[:,:,0].ravel(), bins = 256, color = 'red', alpha = 0.5) 

_ = plt.hist(image[:,:,1].ravel(), bins = 256, color = 'green', alpha = 0.5)

_ = plt.hist(image[:,:,2].ravel(), bins = 256, color = 'blue', alpha = 0.5)

_ = plt.xlabel('Intensity Values')

_ = plt.ylabel('Count')

_ = plt.legend(['red_channel','green_channel','blue_channel'])

plt.show()


