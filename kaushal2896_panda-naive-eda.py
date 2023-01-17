import os

import openslide

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt
!ls /kaggle/input/prostate-cancer-grade-assessment
BASE_DIR = '/kaggle/input/prostate-cancer-grade-assessment'
%%time

train_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))

test_df = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

sample_sub_df = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))
print(f'Number of training images: {len(os.listdir(os.path.join(BASE_DIR, "train_images")))}')

print(f'Number of segmentation masks for training: {len(os.listdir(os.path.join(BASE_DIR, "train_label_masks")))}')
train_df.head()
test_df.head()
sample_sub_df.head()
print(f'Size of train_df: {train_df.shape}')

print(f'Size of test_df: {test_df.shape}')

print(f'Size of sample_sub_df: {sample_sub_df.shape}')
sns.set(rc={'figure.figsize':(11,8)})

sns.set(style="whitegrid")
np.sort(pd.unique(train_df['isup_grade']))
ax = sns.barplot(np.sort(pd.unique(train_df['isup_grade'])), train_df['isup_grade'].value_counts().sort_values(ascending=False))

ax.set(xlabel='ISUP Grades', ylabel='# of records', title='ISUP Grades vs. # of records')

plt.show()
train_df['data_provider'].value_counts()
ax = sns.barplot(np.sort(pd.unique(train_df['data_provider'])), train_df['data_provider'].value_counts().sort_values(ascending=False))

ax.set(xlabel='Data Providers', ylabel='# of records', title='Data Providers vs. # of records')

plt.show()
counts_karolinska = train_df[train_df['data_provider'] == 'karolinska']['isup_grade'].value_counts(ascending=False)

counts_radboud = train_df[~(train_df['data_provider'] == 'karolinska')]['isup_grade'].value_counts()



karolinska_df = pd.DataFrame({

    '# of records': counts_karolinska,

    'isup_grades': np.sort(pd.unique(train_df[train_df['data_provider'] == 'karolinska']['isup_grade'])),

    'data_provider': 'karolinska'

})

radboud_df = pd.DataFrame({

    '# of records': counts_radboud,

    'isup_grades': np.sort(pd.unique(train_df[~(train_df['data_provider'] == 'karolinska')]['isup_grade'])),

    'data_provider': 'radboud'

})

sns.factorplot(x='isup_grades', y='# of records', hue='data_provider', data=pd.concat([karolinska_df, radboud_df], ignore_index=True), kind='bar', height=7, aspect=1.5)

plt.show()
# Visualize few samples of current training dataset

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 20))

count=0

for row in ax:

    for col in row:

        img = os.path.join(BASE_DIR, 'train_images', f'{train_df["image_id"].iloc[count]}.tiff')

        img = openslide.OpenSlide(img)

        patch = img.read_region((0, 0), 2, img.level_dimensions[-1])

        col.title.set_text(f'Source: {train_df["data_provider"].iloc[count]} \n ISUP grade: {train_df["isup_grade"].iloc[count]} \n gleason score: {train_df["gleason_score"].iloc[count]}')

        col.grid(False)

        col.set_xticks([])

        col.set_yticks([])

        col.imshow(patch)

        count += 1

plt.show()
def print_mask_details(slide, center='radboud', show_thumbnail=True, max_size=(400,400)):

    """Print some basic information about a slide"""



    if center not in ['radboud', 'karolinska']:

        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")



    # Generate a small image thumbnail

    if show_thumbnail:

        # Read in the mask data from the highest level

        # We cannot use thumbnail() here because we need to load the raw label data.

        mask_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])

        # Mask data is present in the R channel

        mask_data = mask_data.split()[0]



        # To show the masks we map the raw label values to RGB values

        preview_palette = np.zeros(shape=768, dtype=int)

        if center == 'radboud':

            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}

            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

        elif center == 'karolinska':

            # Mapping: {0: background, 1: benign, 2: cancer}

            preview_palette[0:9] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0]) * 255).astype(int)

        mask_data.putpalette(data=preview_palette.tolist())

        mask_data = mask_data.convert(mode='RGB')

        mask_data.thumbnail(size=max_size, resample=0)

        return mask_data
fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 20))

count=0

for row in ax:

    for col in row:

        mask = os.path.join(BASE_DIR, 'train_label_masks', f'{train_df["image_id"].iloc[count]}_mask.tiff')

        mask = openslide.OpenSlide(mask)

        mask = print_mask_details(mask, center='radboud')

        col.imshow(mask)

        col.title.set_text(f'Source: {train_df["data_provider"].iloc[count]} \n ISUP grade: {train_df["isup_grade"].iloc[count]} \n gleason score: {train_df["gleason_score"].iloc[count]}')

        col.grid(False)

        col.set_xticks([])

        col.set_yticks([])

        mask.close()

        count += 1

plt.show()
def plot_patch(img, x, y, level=0, width=512, height=512):

    biopsy = openslide.OpenSlide(os.path.join(BASE_DIR, 'train_images', img))

    region = biopsy.read_region((x, y), level, (width, height))

    display(region)
plot_patch('00928370e2dfeb8a507667ef1d4efcbb.tiff', 5150, 21000)
plot_patch('0005f7aaab2800f6170c399693a96917.tiff', 6000, 18000)
plot_patch('0018ae58b01bdadc8e347995b69f99aa.tiff', 1500, 6000)
def plot_biopsy_grid(df):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

    count=0

    for row in ax:

        for col in row:

            img = os.path.join(BASE_DIR, 'train_images', f'{df["image_id"].iloc[count]}.tiff')

            img = openslide.OpenSlide(img)

            patch = img.read_region((0, 0), 2, img.level_dimensions[-1])

            col.title.set_text(f'Source: {df["data_provider"].iloc[count]} \n ISUP grade: {df["isup_grade"].iloc[count]} \n gleason score: {df["gleason_score"].iloc[count]}')

            col.grid(False)

            col.set_xticks([])

            col.set_yticks([])

            col.imshow(patch)

            count += 1

    plt.show()
plot_biopsy_grid(train_df[train_df['isup_grade'] == 0][:4])
plot_biopsy_grid(train_df[train_df['isup_grade'] == 1][:4])
plot_biopsy_grid(train_df[train_df['isup_grade'] == 2][:4])
plot_biopsy_grid(train_df[train_df['isup_grade'] == 3][:4])
plot_biopsy_grid(train_df[train_df['isup_grade'] == 4][:4])
plot_biopsy_grid(train_df[train_df['isup_grade'] == 5][:4])