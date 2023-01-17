import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import skimage.io # Loading images

import cv2 # Resizing images

from tqdm.notebook import tqdm # Visualizing progress



import matplotlib.pyplot as plt

import matplotlib.colors



import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



PATH = "../input/prostate-cancer-grade-assessment/"



df_train = pd.read_csv(f'{PATH}train.csv')

df_test = pd.read_csv(f'{PATH}test.csv')



df_train.head().style.set_caption('Quick Overview of train.csv')
print(f"Number of training data: {len(df_train)}\n")



print(f"Unique data_providers: {df_train.data_provider.unique()}\n")

print(f"Unique isup_grade: {df_train.isup_grade.unique()}\n")

print(f"Unique gleason_score: {df_train.gleason_score.unique()}\n")



print(f"Missing data:\n{df_train.isna().any()}\n")



masks = os.listdir(PATH + 'train_label_masks/')

images = os.listdir(PATH + 'train_images/')



df_masks = pd.Series(masks).to_frame()

df_masks.columns = ['mask_file_name']

df_masks['image_id'] = df_masks.mask_file_name.apply(lambda x: x.split('_')[0])

df_train = pd.merge(df_train, df_masks, on='image_id', how='outer')

del df_masks

print(f"There are {len(df_train[df_train.mask_file_name.isna()])} images without a mask.")
print(f"Train data shape before reduction: {len(df_train)}")

df_train_red = df_train[~df_train.mask_file_name.isna()]

print(f"Train data shape after reduction: {len(df_train_red)}")



no_masks = df_train[df_train.mask_file_name.isna()][['image_id']]

no_masks['Suspicious_because'] = 'No Mask'
df_train_red.groupby('isup_grade').gleason_score.unique().to_frame().style.set_caption('Mapping of ISUP Grade to Gleason Score')
df_train_red[(df_train_red.isup_grade == 2) & (df_train_red.gleason_score != '3+4')]
providers = df_train_red.data_provider.unique()



fig = plt.figure(figsize=(6,4))

ax = sns.countplot(x="isup_grade", hue="data_provider", data=df_train_red)

plt.title("ISUP Grade Count by Data Provider", fontsize=14)

plt.xlabel("ISUP Grade", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.show()
df_train_red["height"] = 0

df_train_red["width"] = 0

df_train_red[0] = 0

df_train_red[1] = 0

df_train_red[2] = 0

df_train_red[3] = 0

df_train_red[4] = 0

df_train_red[5] = 0



def get_image_data(row):

    biopsy = skimage.io.MultiImage(PATH + 'train_label_masks/' + row.image_id + '_mask.tiff')

    temp = biopsy[-1][:, :, 0]

    counts = pd.Series(temp.reshape(-1)).value_counts()

    row.height = temp.shape[0]

    row.width = temp.shape[1]

    row.update(counts)

    return row



df_train_red = df_train_red.apply(lambda row: get_image_data(row), axis=1)



df_train_red['pixels'] = df_train_red.height * df_train_red.width

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))



"""

Inspired by something similiar I saw here https://www.kaggle.com/dhananjay3/panda-eda-all-you-need-to-know

"""

sns.scatterplot(data=df_train_red, x='width', y='height', marker='.',hue='data_provider', ax=ax1)

ax1.set_title("Image Sizes by Data Provider", fontsize=14)

ax1.set_xlabel("Image Width", fontsize=14)

ax1.set_ylabel("Image Height", fontsize=14)



sns.kdeplot(df_train_red[df_train_red.data_provider == 'karolinska'].pixels, label='karolinska', ax=ax2)

sns.kdeplot(df_train_red[df_train_red.data_provider == 'radboud'].pixels, label= 'radboud', ax=ax2)



ax2.set_title("Image Sizes by Data Provider", fontsize=14)

ax2.set_ylabel("Pixels per Image", fontsize=14)

plt.show()
empty_masks = df_train_red[(df_train_red[1] == 0) & (df_train_red[2] ==0)& (df_train_red[3] ==0) & (df_train_red[4] ==0) & (df_train_red[5] ==0)]

print(f"There are {len(empty_masks)} masks that only contain background pixels.")

empty_masks[['image_id', 'data_provider', 'isup_grade', 'gleason_score', 0, 1, 2, 3, 4, 5]]
for i in empty_masks.image_id:

    biopsy = skimage.io.MultiImage(PATH + 'train_label_masks/' + i + '_mask.tiff')

    # Check whether all three channels are empty

    no_mask = True

    for j in range(3): 

        if biopsy[-1][:,:, j].max() > 0:

            print(f"Found mask for image {i} in channel {j}")

            no_mask = False

        

    if no_mask == True:

        print(f"Couldn't find mask for image {i} in other channels.")

        

empty_masks = empty_masks[['image_id']]

empty_masks['Suspicious_because'] = 'Background only'
df_train_red[(df_train_red.data_provider == 'karolinska') & (df_train_red.isup_grade > 0) & (df_train_red[2] ==0)][['image_id', 'data_provider', 'isup_grade', 'gleason_score', 0, 1, 2, 3, 4, 5]].style.set_caption('Suspicious Masks provided by Karolinska')
no_cancerous_tissue = df_train_red[(df_train_red.data_provider == 'radboud') & (df_train_red.isup_grade > 0) & (df_train_red[3] ==0) & (df_train_red[4] ==0) & (df_train_red[5] ==0)]

no_cancerous_tissue[['image_id', 'data_provider', 'isup_grade', 'gleason_score', 0, 1, 2, 3, 4, 5]].style.set_caption('Suspicious Masks provided by Radboud')
print(f"Train data shape before second reduction: {len(df_train_red)}")

df_train_red = df_train_red[(~df_train_red.image_id.isin(empty_masks.image_id)) & (~df_train_red.image_id.isin(no_cancerous_tissue.image_id))]

print(f"Train data shape after second reduction: {len(df_train_red)}")



no_cancerous_tissue = no_cancerous_tissue[['image_id']]

no_cancerous_tissue['Suspicious_because'] = 'No cancerous tissue but ISUP Grade > 0'
df_train_red['tissue'] = df_train_red[1] + df_train_red[2] + df_train_red[3] + df_train_red[4] + df_train_red[5]



fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))





karolinska = df_train_red[df_train_red.data_provider == 'karolinska'].groupby('isup_grade')[[1, 2, 'tissue']].mean()

karolinska[1] = karolinska[1] /karolinska['tissue']*100

karolinska[2] = karolinska[2] /karolinska['tissue'] *100

karolinska = karolinska.drop(['tissue'], axis=1)



karolinska.plot(kind='bar', stacked=True, ax=ax1, cmap = matplotlib.colors.ListedColormap(['green','red']))

ax1.set_title("Tissue Labels by Karolinska", fontsize=14)

ax1.set_xlabel("ISUP Grade", fontsize=14)

ax1.set_ylabel("Percentage of Labeled Tissue", fontsize=14)





radboud = df_train_red[df_train_red.data_provider == 'radboud'].groupby('isup_grade')[[1, 2, 3, 4, 5, 'tissue']].mean()

radboud[1] = radboud[1] /radboud['tissue']*100

radboud[2] = radboud[2] /radboud['tissue'] *100

radboud[3] = radboud[3] /radboud['tissue'] *100

radboud[4] = radboud[4] /radboud['tissue'] *100

radboud[5] = radboud[5] /radboud['tissue'] *100



radboud = radboud.drop(['tissue'], axis=1)



radboud.plot(kind='bar', stacked=True, ax=ax2, cmap = matplotlib.colors.ListedColormap(['lightgrey', 'green', 'orange', 'red', 'darkred']))

ax2.set_title("Tissue Labels by Radboud", fontsize=14)

ax2.set_xlabel("ISUP Grade", fontsize=14)

ax2.set_ylabel("Percentage of Labeled Tissue", fontsize=14)



plt.show()
def load_and_resize_image(img_id):

    """

    Edited from https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data

    """

    path = PATH + 'train_images/' + img_id + '.tiff'

    biopsy = skimage.io.MultiImage(path)

    return cv2.resize(biopsy[-1], (512, 512))



def load_and_resize_mask(img_id):

    """

    Edited from https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data

    """

    path = PATH + 'train_label_masks/' + img_id + '_mask.tiff'

    biopsy = skimage.io.MultiImage(path)

    return cv2.resize(biopsy[-1], (512, 512))[:,:,0]
labels = []

for grade in range(df_train_red.isup_grade.nunique()):

    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))



    for i, row in enumerate(ax):

        idx = i//2

        temp = df_train_red[(df_train_red.isup_grade == grade) & (df_train_red.data_provider == providers[idx])].image_id.head(4).reset_index(drop=True)

        

        if i%2 < 1:

            labels.append(f'{providers[idx]} (image)')

            for j, col in enumerate(row):

                col.imshow(load_and_resize_image(temp[j]))

        else:

            labels.append(f'{providers[idx]} (mask)')

            for j, col in enumerate(row):

                if providers[idx] == 'radboud':

                    col.imshow(load_and_resize_mask(temp[j]), 

                               cmap = matplotlib.colors.ListedColormap(['white', 'lightgrey', 'green', 'orange', 'red', 'darkred']), 

                               norm = matplotlib.colors.Normalize(vmin=0, vmax=5, clip=True))

                else:

                    col.imshow(load_and_resize_mask(temp[j]), 

                           cmap = matplotlib.colors.ListedColormap(['white', 'green', 'red']), 

                           norm = matplotlib.colors.Normalize(vmin=0, vmax=2, clip=True))



    for row, r in zip(ax[:,0], labels):

        row.set_ylabel(r, rotation=90, size='large', fontsize=14)



    plt.suptitle(f'ISUP Grade {grade}', fontsize=14)

    plt.show()
#import matplotlib.cm as cm

common_cmap = matplotlib.colors.ListedColormap(['white', 'green', 'red'])

norm = matplotlib.colors.Normalize(vmin=0, vmax=2, clip=True)

#mapper = cm.ScalarMappable(norm=norm, cmap=common_cmap)



common_mask_dict = { 0 : 0, #background (non tissue) or unknown

                    1 : 1, # benign tissue (stroma and epithelium combined)

                    2 : 1, # benign tissue (stroma and epithelium combined),

                    3 : 2, # cancerous tissue (stroma and epithelium combined)

                    4 : 2, # cancerous tissue (stroma and epithelium combined)

                    5 : 2, # cancerous tissue (stroma and epithelium combined)

                   }

def load_and_resize_mask_with_common_mask(img_id):

    path = PATH + 'train_label_masks/' + img_id + '_mask.tiff'

    biopsy = skimage.io.MultiImage(path)

    mask = cv2.resize(biopsy[-1], (512, 512))[:,:,0]

    

    return np.array([common_mask_dict[letter] for letter in mask.reshape(-1)]).reshape(512, 512)

   

labels = []

for grade in range(df_train_red.isup_grade.nunique()):

    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))



    for i, row in enumerate(ax):

        idx = i//2

        temp = df_train_red[(df_train_red.isup_grade == grade) & (df_train_red.data_provider == providers[idx])].image_id.head(4).reset_index(drop=True)

        

        if i%2 < 1:

            labels.append(f'{providers[idx]} (image)')

            for j, col in enumerate(row):

                col.imshow(load_and_resize_image(temp[j]))

        else:

            labels.append(f'{providers[idx]} (mask)')

            for j, col in enumerate(row):

                if providers[idx] == 'radboud':

                    col.imshow(load_and_resize_mask_with_common_mask(temp[j]), cmap = common_cmap,norm=norm)

                else: 

                    col.imshow(load_and_resize_mask(temp[j]), cmap = common_cmap,norm=norm)

    for row, r in zip(ax[:,0], labels):

        row.set_ylabel(r, rotation=90, size='large', fontsize=14)



    plt.suptitle(f'ISUP Grade {grade}', fontsize=14)

    plt.show()
suspicious_test_cases = no_masks.reset_index(drop=True)

suspicious_test_cases = suspicious_test_cases.append(empty_masks.reset_index(drop=True))

suspicious_test_cases = suspicious_test_cases.append(no_cancerous_tissue.reset_index(drop=True))

suspicious_test_cases.to_csv("suspicious_test_cases.csv",index=False)

suspicious_test_cases
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))



sns.countplot(df_train.data_provider, ax=ax[0])



ax[0].set_title("Image Sizes by Data Provider", fontsize=14)

ax[0].set_title("Orginial Train Set (Wtih Suspicious Data)", fontsize=14)

ax[0].set_xlabel("Data Provider", fontsize=14)

ax[0].set_ylabel("Number of Image in Set", fontsize=14)



sns.countplot(df_train_red.data_provider, ax=ax[1])

ax[1].set_title("Reduced Train Set (Without Suspicious Data)", fontsize=14)

ax[1].set_xlabel("Data Provider", fontsize=14)

ax[1].set_ylabel("Number of Image in Set", fontsize=14)



sns.countplot(df_test.data_provider, ax=ax[2], order=df_test.data_provider.value_counts().sort_values(ascending=True).index)

ax[2].set_title("Test Set", fontsize=14)

ax[2].set_xlabel("Data Provider", fontsize=14)

ax[2].set_ylabel("Number of Image in Set", fontsize=14)

ax[2].set_ylim([0, 20])

plt.show()



sample_submission = pd.read_csv(f'{PATH}sample_submission.csv')

sample_submission.style.set_caption('Sample Submission')