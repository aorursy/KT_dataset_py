import os
import numpy as np
import pandas as pd

PATH = '/kaggle/input/siim-isic-melanoma-classification'
print(os.listdir(PATH))
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
benign_df = train_df[train_df['benign_malignant']=='benign']
malignant_df = train_df[train_df['benign_malignant']=='malignant']
train_df.head()
train_df.info()
num_training = len(train_df)
num_benign = len(train_df[train_df['benign_malignant']=='benign'])
num_malignant = len(train_df[train_df['benign_malignant']=='malignant'])

print("Total number of records :", len(train_df))
print("Number of Benign records :", len(train_df[train_df['benign_malignant']=='benign']))
print("Number of Malignant records :", len(train_df[train_df['benign_malignant']=='malignant']))
print(f"Percentage of Benign records : {num_benign/num_training:.2f}")
print(f"Percentage of Malignant records : {num_malignant/num_training:.2f}")
# Plots of distrubution of sex
fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0]=train_df['sex'].dropna().value_counts().plot(kind='bar', ax=ax[0])
ax[1]=train_df[(train_df['benign_malignant']=='benign')]['sex'].dropna().value_counts().plot(kind='bar', ax=ax[1])
ax[2]=train_df[(train_df['benign_malignant']=='malignant')]['sex'].dropna().value_counts().plot(kind='bar', ax=ax[2])

for ax_ in ax:
    for p in ax_.patches:
        ax_.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        
ax[0].title.set_text('Train Set Sex Distribuiton')
ax[1].title.set_text('Benign Subset Sex Distribuiton')
ax[2].title.set_text('Malignant Subset Sex Distribuiton')

print("Number of missing rows : ", len(train_df['sex'])-train_df['sex'].count())
#Distribution of age for each individually
fig, ax = plt.subplots(1,3, figsize=(15,5))

ax[0]=train_df['age_approx'].dropna().plot(kind='density', ax=ax[0])
ax[1]=train_df['age_approx'].dropna().plot(kind='hist', ax=ax[1])
ax[2]=train_df['age_approx'].dropna().plot(kind='box', ax=ax[2])

for p in ax[1].patches:
        ax[1].annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))
        
fig.suptitle('Age Distribution')
print("Number of missing rows : ", len(train_df['age_approx'])-train_df['age_approx'].count())
fig, ax = plt.subplots(1,3, figsize=(15,5))
_ = train_df[train_df["target"]==0].age_approx.hist(bins=20, ax=ax[0])
_ = train_df[train_df["target"]==1].age_approx.hist(bins=20, ax=ax[1])
_ = train_df.groupby("target").age_approx.hist(bins=20, alpha=1, ax=ax[2])
_ = ax[0].set_title('Age Distribution for Benign Cases')
_ = ax[1].set_title('Age Distribution for Malignant Cases')
_ = ax[2].set_title('Age Distribution Comparision')
print("Possible values of age :",np.sort(train_df.age_approx.unique()))
ax = train_df['diagnosis'].value_counts().plot(kind='bar')
for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))
        
ax.set_title('Diagnosis Distribution')
print("Number of unknown diagnosis : ", len(train_df[train_df['diagnosis']=='unknown']))
fig, ax = plt.subplots(1,3, figsize=(15,5))

ax[0]=train_df['anatom_site_general_challenge'].value_counts().dropna().plot(kind='bar', ax=ax[0])
ax[1]=benign_df['anatom_site_general_challenge'].value_counts().dropna().plot(kind='bar', ax=ax[1], colormap='plasma')
ax[2]=malignant_df['anatom_site_general_challenge'].value_counts().dropna().plot(kind='bar', ax=ax[2])

for ax_ in ax:
    for p in ax_.patches:
            ax_.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))

_ = ax[0].set_title('Train Dataset Distribution')
_ = ax[1].set_title('Benign Set Distribution')
_ = ax[2].set_title('Malignant Set Distribution')
_ = fig.suptitle('Anatom Sight Distribution', fontweight='black', fontsize=15)
print("Number of missing rows : ", len(train_df['anatom_site_general_challenge'])-train_df['anatom_site_general_challenge'].count())
print(f"Total number of records : {len(train_df)}\n\
Number of patients : {len(train_df.patient_id.unique())}")
overlap = len(set(train_df.patient_id.unique()).intersection(set(test_df.patient_id.unique())))
print(f"Number of patients common in train and test set = {overlap}")
TRAIN_IMAGE_PATH = os.path.join(PATH,'jpeg','train') 
TRAIN_IMAGES_LIST = os.listdir(TRAIN_IMAGE_PATH)
fig, ax = plt.subplots(4,4,figsize=(15,15))
 
for i, row in enumerate(ax):
    for j, cell in enumerate(row):
        idx = np.random.randint(len(TRAIN_IMAGES_LIST))
        ax[i,j].imshow(plt.imread(os.path.join(TRAIN_IMAGE_PATH, TRAIN_IMAGES_LIST[idx])))
        ax[i,j].axis('off')
#         print(f"Reading image {TRAIN_IMAGES_LIST[idx]}")
fig, ax = plt.subplots(4,4,figsize=(15,15))
BENIGN_IMAGES_LIST = list(benign_df['image_name'])
for i, row in enumerate(ax):
    for j, cell in enumerate(row):
        idx = np.random.randint(len(BENIGN_IMAGES_LIST))
        ax[i,j].imshow(plt.imread(os.path.join(TRAIN_IMAGE_PATH, BENIGN_IMAGES_LIST[idx]+'.jpg')))
        ax[i,j].axis('off')
#         print(f"Reading image {TRAIN_IMAGES_LIST[idx]}")
fig, ax = plt.subplots(4,4,figsize=(15,15))
MALIGNANT_IMAGES_LIST = list(malignant_df['image_name'])
for i, row in enumerate(ax):
    for j, cell in enumerate(row):
        idx = np.random.randint(len(MALIGNANT_IMAGES_LIST))
        ax[i,j].imshow(plt.imread(os.path.join(TRAIN_IMAGE_PATH, MALIGNANT_IMAGES_LIST[idx]+'.jpg')))
        ax[i,j].axis('off')
#         print(f"Reading image {TRAIN_IMAGES_LIST[idx]}")
benign_image_sample = plt.imread(os.path.join(TRAIN_IMAGE_PATH, BENIGN_IMAGES_LIST[np.random.randint(len(BENIGN_IMAGES_LIST))]+'.jpg'))

fig, ax = plt.subplots(1,2,figsize=(15,5))
ax[0].imshow(benign_image_sample)
ax[0].set_xlabel(f"Image dimensions : {benign_image_sample.shape[:2]}")
ax[0].set_xticks([])
ax[0].set_yticks([])

_ = ax[1].hist(benign_image_sample[:,:, 0].ravel(), bins=256, color='red', alpha=0.5)
_ = ax[1].hist(benign_image_sample[:,:, 1].ravel(), bins=256, color='green', alpha=0.7)
_ = ax[1].hist(benign_image_sample[:,:, 2].ravel(), bins=256, color='blue', alpha=0.5)
_ = ax[1].set_xlabel('Pixel Intensities')
_ = ax[1].set_ylabel('Pixel Counts')
_ = ax[1].legend(['Red Channel', 'Green Channel', 'Blue Channel'])
_ = plt.suptitle("Pixel Intensities for Benign Image")
malignant_image_sample = plt.imread(os.path.join(TRAIN_IMAGE_PATH, MALIGNANT_IMAGES_LIST[np.random.randint(len(MALIGNANT_IMAGES_LIST))]+'.jpg'))
fig, ax = plt.subplots(1,2,figsize=(15,5))
ax[0].imshow(malignant_image_sample)
ax[0].set_xlabel(f"Image dimensions : {malignant_image_sample.shape[:2]}")
ax[0].set_xticks([])
ax[0].set_yticks([])

_ = ax[1].hist(malignant_image_sample[:,:, 0].ravel(), bins=256, color='red', alpha=0.5)
_ = ax[1].hist(malignant_image_sample[:,:, 1].ravel(), bins=256, color='green', alpha=0.7)
_ = ax[1].hist(malignant_image_sample[:,:, 2].ravel(), bins=256, color='blue', alpha=0.5)
_ = ax[1].set_xlabel('Pixel Intensities')
_ = ax[1].set_ylabel('Pixel Counts')
_ = ax[1].legend(['Red Channel', 'Green Channel', 'Blue Channel'])
_ = plt.suptitle("Pixel Intensities for Malignant Image")