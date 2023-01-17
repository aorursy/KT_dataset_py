import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

import pydicom

import os

import glob

import imageio

from IPython.display import Image



plt.rcParams["figure.figsize"] = (10,5)
base_dir = '../input/osic-pulmonary-fibrosis-progression/'

os.listdir(base_dir)
train_path = base_dir + 'train/'

test_path = base_dir + 'test/'
train_df = pd.read_csv(base_dir+'train.csv')

train_df.head()
train_df.info()
train_df.describe()
sb.countplot(data = train_df, x="Sex")

plt.title("Sex distribution")
sb.boxplot(data=train_df, x = 'Sex', y = 'Age')

plt.title('Sex distribution based on Age')
sb.countplot(train_df['Age'])

plt.title('Age distribution')
sb.kdeplot(train_df.loc[train_df['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)

sb.kdeplot(train_df.loc[train_df['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)

plt.xlabel('Age (years)'); plt.ylabel('Density') 

plt.title('Distribution of Ages')
sb.countplot(data = train_df, x="SmokingStatus")

plt.title("Smoking status distribution")
sb.countplot(data = train_df, x="SmokingStatus", hue='Sex')

plt.title('Smoking Status distribution based on Sex')




sb.scatterplot(data = train_df, x="FVC", y="Percent", hue='Age')

plt.title('FVC vs Percent')
sb.scatterplot(data = train_df, x="FVC", y="Age", hue='Sex')

plt.title('FVC vs Age')
sb.scatterplot(data = train_df, x="FVC", y="Weeks", hue='SmokingStatus')

plt.title('FVC vs Weeks')
sb.distplot(train_df['Percent'])

plt.title('Percent distribution')
sb.violinplot(data=train_df, x='Percent', y='SmokingStatus', hue = 'Sex')

plt.title('Percent vs Smoking status')
sb.scatterplot(data = train_df, x="Weeks", y="Age", hue = "Sex")

plt.title('Weeks vs Age')
patient_id_1 = train_df.Patient[0]

patient_1 = train_df[train_df.Patient == patient_id_1]

patient_1
patient_1.plot(x='Weeks',y='FVC')

plt.title('Weeks vs FVC for 1 patient')
patient_1.plot(x='Weeks',y='Percent')

plt.title('Weeks vs Percent for 1 patient')
sb.heatmap(train_df.corr(), cmap = 'RdYlBu_r')

plt.title('Correlation Matrix')
patient_1_path = train_path + patient_id_1 +'/'

img_paths = [

        f for f in glob.glob(

            os.path.join(patient_1_path, '**')

        )]

img_paths = sorted(img_paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
fig, axs = plt.subplots(5,6, figsize = (12,12))

axs = axs.flatten()

for image_path,axis in zip(img_paths,axs):

    img = pydicom.dcmread(image_path)

    axis.imshow(img.pixel_array)

fig.suptitle('Lung Images over Weeks', fontweight='bold')
images = []

for image_path in img_paths:

    images.append(pydicom.dcmread(image_path).pixel_array)

imageio.mimsave("/tmp/gif.gif", images, duration=0.0001)

Image(filename="/tmp/gif.gif", format='png')

