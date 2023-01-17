import pandas as pd

import numpy as np

from glob import glob

import cv2

from skimage import io

from tqdm import tqdm

import seaborn as sns
!mkdir '1024x1024-dataset-melanoma'

df_train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
df_gt = pd.read_csv('../input/isic-2019/ISIC_2019_Training_GroundTruth.csv')

image_id = df_gt.iloc[25]['image']

image = cv2.imread(f'../input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/{image_id}.jpg', cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

io.imshow(image);
df_downsampled = df_gt[df_gt['image'].str.contains('downsampled')]

df_downsampled.shape[0]
print('[ALL]:', df_gt.shape[0])

print('[∩ isic2020]:', len(set(df_train['image_name'].values).intersection(df_gt['image'].values)))

print('[downsampled isic2019 ∩ isic2020]:', len(set(df_train['image_name'].values).intersection([

    image_id[:-12] for image_id in df_downsampled['image'].values

])))

print('[downsampled isic2019 ∩ isic2019]:', len(set(df_gt['image'].values).intersection([

    image_id[:-12] for image_id in df_downsampled['image'].values

])))
paths = glob('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/*/*/*.jpg')

image = cv2.imread(paths[777], cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

io.imshow(image);
image_ids = [path.split('/')[-1][:-4] for path in paths]

print('[ALL]:', len(image_ids))

print('[∩ isic2020]:', len(set(image_ids).intersection(df_train['image_name'].values)))

print('[∩ isic2019]:', len(set(image_ids).intersection(df_gt['image'].values)))

print('[∩ isic2019 downsampled]:', len(set(image_ids).intersection([image_id[:-12] for image_id in df_gt[df_gt['image'].str.contains('downsampled')]['image'].values])))
df_meta = pd.read_csv('../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')
image_id = df_meta.iloc[777]['image_id']

image = cv2.imread(f'../input/skin-cancer-mnist-ham10000/HAM10000_images_part_1/{image_id}.jpg', cv2.IMREAD_COLOR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

io.imshow(image);
print('[ALL]:', df_meta.shape[0])

print('[∩ isic2020]:', len(set(df_meta['image_id'].values).intersection(df_train['image_name'].values)))

print('[∩ isic2019]:', len(set(df_meta['image_id'].values).intersection(df_gt['image'].values)))

print('[∩ slatmd]:', len(set(df_meta['image_id'].values).intersection(image_ids)))
NEED_IMAGE_SAVE = False
dataset = {

    'patient_id' : [],

    'image_id': [],

    'target': [],

    'source': [],

    'sex': [],

    'age_approx': [],

    'anatom_site_general_challenge': [],

}



# isic2020

df_train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv', index_col='image_name')

for image_id, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):

    if image_id in dataset['image_id']:

        continue

    dataset['patient_id'].append(row['patient_id'])

    dataset['image_id'].append(image_id)

    dataset['target'].append(row['target'])

    dataset['source'].append('ISIC20')

    dataset['sex'].append(row['sex'])

    dataset['age_approx'].append(row['age_approx'])

    dataset['anatom_site_general_challenge'].append(row['anatom_site_general_challenge'])



    if NEED_IMAGE_SAVE:

        image = cv2.imread(f'../input/siim-isic-melanoma-classification/jpeg/train/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (1024, 1024), cv2.INTER_AREA)

        cv2.imwrite(f'./1024x1024-dataset-melanoma/{image_id}.jpg', image)



# isic2019

df_gt = pd.read_csv('../input/isic-2019/ISIC_2019_Training_GroundTruth.csv', index_col='image')

df_meta = pd.read_csv('../input/isic-2019/ISIC_2019_Training_Metadata.csv', index_col='image')

for image_id, row in tqdm(df_meta.iterrows(), total=df_meta.shape[0]):

    if image_id in dataset['image_id']:

        continue



    dataset['patient_id'].append(row['lesion_id'])

    dataset['image_id'].append(image_id)

    dataset['target'].append(int(df_gt.loc[image_id]['MEL']))

    dataset['source'].append('ISIC19')

    dataset['sex'].append(row['sex'])

    dataset['age_approx'].append(row['age_approx'])

    dataset['anatom_site_general_challenge'].append(

        {'anterior torso': 'torso', 'posterior torso': 'torso'}.get(row['anatom_site_general'], row['anatom_site_general'])

    )

    

    if NEED_IMAGE_SAVE:

        image = cv2.imread(f'../input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.resize(image, (1024, 1024), cv2.INTER_AREA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f'./1024x1024-dataset-melanoma/{image_id}.jpg', image)



    

dataset = pd.DataFrame(dataset)    
dataset.head()
dataset.to_csv('marking.csv', index=False)