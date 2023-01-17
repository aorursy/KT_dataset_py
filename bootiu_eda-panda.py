%matplotlib inline

import os, glob, tifffile, cv2, gc

import openslide

import skimage.io

import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

import seaborn as sns



from tqdm import tqdm_notebook as tqdm
os.listdir('../input/prostate-cancer-grade-assessment')
data_dir = '../input/prostate-cancer-grade-assessment'

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
print(train.shape)

train.head()
fig = plt.figure(figsize=(12, 5), facecolor='w')

sns.countplot(train['data_provider'])

plt.title('Data Provider')

plt.show()
fig = plt.figure(figsize=(12, 5), facecolor='w')

sns.countplot(train['isup_grade'])

plt.title('ISUP Grade (Target)')

plt.show()
fig = plt.figure(figsize=(12, 5), facecolor='w')

sns.countplot(train['gleason_score'])

plt.title('Gleason Score')

plt.show()
score_num = train['gleason_score'].nunique()

fig, axes = plt.subplots(ncols=4, nrows=int(np.ceil(score_num / 4)), figsize=(16, 10), facecolor='w')



for score, ax in zip(train['gleason_score'].unique(), axes.ravel()):

    temp = train[train['gleason_score'] == score]

    sns.countplot(temp['isup_grade'], ax=ax)

    ax.set_title(score)

    

plt.tight_layout()

plt.show()
# isup_grade = 2 and gleason_score = "4+3"

a = train[train['isup_grade'] == 2]

a[a['gleason_score'] == '4+3']
train[train['gleason_score'] == 'negative']['data_provider'].unique()
train[train['gleason_score'] == '0+0']['data_provider'].unique()
# Modify label

train.loc[train['gleason_score'] == 'negative', 'gleason_score'] = '0+0'

train.loc[train['isup_grade'] == 2, 'gleason_score'] = '3+4'
score_num = train['isup_grade'].nunique()

fig, axes = plt.subplots(ncols=3, nrows=int(np.ceil(score_num / 4)), figsize=(16, 10), facecolor='w')



for score, ax in zip(sorted(train['isup_grade'].unique()), axes.ravel()):

    temp = train[train['isup_grade'] == score]

    sns.countplot(temp['gleason_score'], ax=ax)

    ax.set_title(score)

    

plt.tight_layout()

plt.show()
train_imgs = glob.glob(os.path.join(data_dir, 'train_images/*.tiff'))

train_imgs = sorted(train_imgs)

print(len(train_imgs))

train_imgs[:3]
train_masks = glob.glob(os.path.join(data_dir, 'train_label_masks/*.tiff'))

train_masks = sorted(train_masks)

print(len(train_masks))

train_masks[:3]
def display_img(img_id, show_img=True, tiff_level=-1):

    img_path = os.path.join(data_dir, 'train_images', f'{img_id}.tiff')

    # Using Openslide

    slide = openslide.OpenSlide(img_path)

    # Set Properties  1: Point   2: Tiff Level   3: Viewing Dimension

    # .level_count -> Get Tiff Level Count

    # .level_dimensions -> Get Tiff Width, Height per Level

    if tiff_level == -1:

        patch = slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[-1])

    else:

        patch = slide.read_region((0, 0), tiff_level, slide.level_dimensions[tiff_level])

    

    if show_img:

        display(patch)

        

    # PIL -> ndarray

    patch = np.asarray(patch)

    # RGBA -> RGB

    if patch.shape[-1] == 4:

        patch = patch[:, :, :3]

        

    slide.close()

    

    return patch





def display_mask(img_id, center, show_img=True, tiff_level=-1):

    assert center in ['radboud', 'karolinska'], "Please Set center=['radboud', 'karolinska']"

    

    img_path = os.path.join(data_dir, 'train_label_masks', f'{img_id}_mask.tiff')

    # Using Openslide

    slide = openslide.OpenSlide(img_path)

    # Set Properties  1: Point   2: Tiff Level   3: Viewing Dimension

    # .level_count -> Get Tiff Level Count

    # .level_dimensions -> Get Tiff Width, Height per Level

    if tiff_level == -1:

        mask_data = slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[-1])

    else:

        mask_data = slide.read_region((0, 0), tiff_level, slide.level_dimensions[tiff_level])

    

    mask_data = mask_data.split()[0]

    # To show the masks we map the raw label values to RGB values

    preview_palette = np.zeros(shape=768, dtype=int)

    if center == 'radboud':

        # Mapping: {0: background(Black), 1: stroma(Dark Gray), 2: benign epithelium(Light Gray), 3: Gleason 3(Pale Yellow), 4: Gleason 4(Orange), 5: Gleason 5(Red)}

        preview_palette[0:18] = (np.array([0, 0, 0, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

    elif center == 'karolinska':

        # Mapping: {0: background(Black), 1: benign(Gray), 2: cancer(Red)}

        preview_palette[0:9] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0]) * 255).astype(int)

    mask_data.putpalette(data=preview_palette.tolist())

    mask_data = mask_data.convert(mode='RGB')

    if show_img:

        display(mask_data)

        

    # PIL -> ndarray

    mask_data = np.asarray(mask_data)

    # RGBA -> RGB

    if mask_data.shape[-1] == 4:

        mask_data = mask_data[:, :, :3]

        

    slide.close()

    

    return mask_data
img = display_img('0018ae58b01bdadc8e347995b69f99aa')
# Mask Data: Those judged to be cancer

mask = display_mask('0018ae58b01bdadc8e347995b69f99aa', 'radboud')
# Mask Data: Those judged not to be cancer

mask = display_mask('001d865e65ef5d2579c190a0e0350d8f', 'karolinska')
train.head()
score_list = train['gleason_score'].unique()

score_list
# Show mask data for each gleason_score (karolinska)



# Extract only images with mask data

mask_ids = glob.glob(os.path.join(data_dir, 'train_label_masks/*.tiff'))

mask_ids = [id.split('/')[4].split('_')[0] for id in mask_ids]

mask_ids[:6]

_train = train[train['image_id'].isin(mask_ids)]



_train = _train[_train['data_provider'] == 'karolinska']



for score in score_list:

    temp = _train[_train['gleason_score'] == score].sample(frac=1.0)

    

    ids = temp.head(6)['image_id'].values

    data_provider = temp.head(6)['data_provider'].values

    

    fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(24, 12), facecolor='w')

    for _id, provider, ax in zip(ids, data_provider, axes.ravel()):

        mask = display_mask(_id, provider, show_img=False)

        ax.imshow(mask)

        ax.set_title('{} - {}'.format(score, _id))

    plt.tight_layout()

    plt.show()
# Show mask data for each gleason_score (radboud)



# Extract only images with mask data

mask_ids = glob.glob(os.path.join(data_dir, 'train_label_masks/*.tiff'))

mask_ids = [id.split('/')[4].split('_')[0] for id in mask_ids]

mask_ids[:6]

_train = train[train['image_id'].isin(mask_ids)]



_train = _train[_train['data_provider'] == 'radboud']



for score in score_list:

    temp = _train[_train['gleason_score'] == score].sample(frac=1.0)

    

    ids = temp.head(6)['image_id'].values

    data_provider = temp.head(6)['data_provider'].values

    

    fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(24, 12), facecolor='w')

    for _id, provider, ax in zip(ids, data_provider, axes.ravel()):

        mask = display_mask(_id, provider, show_img=False)

        ax.imshow(mask)

        ax.set_title('{} - {}'.format(score, _id))

    plt.tight_layout()

    plt.show()
score_num = train['isup_grade'].nunique()

fig, axes = plt.subplots(ncols=3, nrows=int(np.ceil(score_num / 4)), figsize=(20, 10), facecolor='w')



for score, ax in zip(sorted(train['isup_grade'].unique()), axes.ravel()):

    temp = train[train['isup_grade'] == score]

    sns.countplot(temp['gleason_score'], ax=ax)

    ax.set_title(score)

    

plt.tight_layout()

plt.show()
def preprocessing_mask(mask):



    # RBG -> gleason_score

    _mask = np.sum(mask, axis=2)



    # 0: Background

    # 1: stroma = 153

    _mask = np.where(_mask==153, 1, _mask)

    # 2: benign epithelium = 306

    _mask = np.where(_mask==306, 1, _mask)  # Healthy

    # 3: Gleason 3 = 688  score=3

    _mask = np.where(_mask==688, 3, _mask)

    # 4: Gleason 4 = 382   score=4

    _mask = np.where(_mask==382, 4, _mask)

    # 5: Gleason 5 = 255   score=5

    _mask = np.where(_mask==255, 5, _mask)



    u, counts = np.unique(_mask, return_counts=True)

    score_dict = {k:v for k, v in zip(u, counts)}

    

    return _mask, score_dict
# Calculate percentage for each gleason_score

score = '3+3'

search_num = 5



temp = train[train['data_provider'] == 'radboud']

temp = temp[temp['gleason_score'] == score]



tar_ids = temp.sample(frac=1.0)['image_id'].values[:search_num]



for id in tar_ids:

    try:

        mask = display_mask(id, 'radboud', show_img=False, tiff_level=0)

    except:

        continue

        

    _, score_dict = preprocessing_mask(mask)

    

    try:

        rate = score_dict[3] / score_dict[1] * 100

    except:

        rate = 0

        

    print('#'*30)

    print('Image_id: ', id)

    print(score_dict)

    print(f'Score 3 / All   Rate: {rate:.3f}%')

    

    del mask

    gc.collect()
# Calculate percentage for each gleason_score

score = '3+4'

search_num = 5



temp = train[train['data_provider'] == 'radboud']

temp = temp[temp['gleason_score'] == score]



tar_ids = temp.sample(frac=1.0)['image_id'].values[:search_num]



for id in tar_ids:

    try:

        mask = display_mask(id, 'radboud', show_img=False, tiff_level=0)

    except:

        continue

        

    _, score_dict = preprocessing_mask(mask)

    

    try:

        rate_3 = score_dict[3] / (score_dict[1] + score_dict[4]) * 100

        rate_4 = score_dict[4] / (score_dict[1] + score_dict[3]) * 100

    except:

        rate_3 = 0

        rate_4 = 0

        

    print('#'*30)

    print('Image_id: ', id)

    print(score_dict)

    print(f'Score 3 / All   Rate: {rate_3:.3f}%')

    print(f'Score 4 / All   Rate: {rate_4:.3f}%')

    

    del mask

    gc.collect()
# Calculate percentage for each gleason_score

score = '4+3'

search_num = 5



temp = train[train['data_provider'] == 'radboud']

temp = temp[temp['gleason_score'] == score]



tar_ids = temp.sample(frac=1.0)['image_id'].values[:search_num]



for id in tar_ids:

    try:

        mask = display_mask(id, 'radboud', show_img=False, tiff_level=0)

    except:

        continue

        

    _, score_dict = preprocessing_mask(mask)

    

    try:

        rate_3 = score_dict[3] / (score_dict[1] + score_dict[4]) * 100

        rate_4 = score_dict[4] / (score_dict[1] + score_dict[3]) * 100

    except:

        rate_3 = 0

        rate_4 = 0

        

    print('#'*30)

    print('Image_id: ', id)

    print(score_dict)

    print(f'Score 3 / All   Rate: {rate_3:.3f}%')

    print(f'Score 4 / All   Rate: {rate_4:.3f}%')

    

    del mask

    gc.collect()
# Calculate percentage for each gleason_score

score = '5+4'

search_num = 5



temp = train[train['data_provider'] == 'radboud']

temp = temp[temp['gleason_score'] == score]



tar_ids = temp.sample(frac=1.0)['image_id'].values[:search_num]



for id in tar_ids:

    try:

        mask = display_mask(id, 'radboud', show_img=False, tiff_level=0)

    except:

        continue

        

    _, score_dict = preprocessing_mask(mask)

    

    try:

        rate_4 = score_dict[4] / (score_dict[1] + score_dict[5]) * 100

        rate_5 = score_dict[5] / (score_dict[1] + score_dict[4]) * 100

    except:

        rate_4 = 0

        rate_5 = 0

        

    print('#'*30)

    print('Image_id: ', id)

    print(score_dict)

    print(f'Score 4 / All   Rate: {rate_4:.3f}%')

    print(f'Score 5 / All   Rate: {rate_5:.3f}%')

    

    del mask

    gc.collect()
# Data Provider == 'radboud'

# Calculate percentage for each gleason_score



ids = []

gscore_list = []

rate_3_list = []

rate_4_list = []

rate_5_list = []

limit = 800



temp = train[train['data_provider'] == 'radboud']

temp = temp.sample(frac=1.0).head(limit)

tar_ids = temp['image_id'].values

gleason_score = temp['gleason_score'].values



for i in tqdm(range(len(temp))):

    id = tar_ids[i]

    gscore = gleason_score[i]

    

    try:

        mask = display_mask(id, 'radboud', show_img=False, tiff_level=0)

    except:

        continue

        

    _, score_dict = preprocessing_mask(mask)

    

    # Excluding background

    del score_dict[0]

    all_pix = np.sum([v for v in score_dict.values()])

    

    # Calculate percentage for each score

    try:

        rate_3 = score_dict[3] / all_pix * 100

    except:

        rate_3 = 0

    

    try:

        rate_4 = score_dict[4] / all_pix * 100

    except:

        rate_4 = 0

        

    try:

        rate_5 = score_dict[5] / all_pix * 100

    except:

        rate_5 = 0

        

    ids.append(id)

    gscore_list.append(gscore)

    rate_3_list.append(rate_3)

    rate_4_list.append(rate_4)

    rate_5_list.append(rate_5)

    

    del mask, score_dict

    gc.collect()

    

rate_res = pd.DataFrame({

    'image_id': ids,

    'gleason_score': gscore_list,

    'rate_3': rate_3_list,

    'rate_4': rate_4_list,

    'rate_5': rate_5_list

})
# ScatterPlot

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8), facecolor='w')

scores = ['3+4', '4+3', '4+5', '5+4', '3+5', '5+3']

for score, ax in zip(scores, axes.ravel()):

    temp = rate_res[rate_res['gleason_score'] == score]

    sns.scatterplot(x='rate_{}'.format(score.split('+')[0]), y='rate_{}'.format(score.split('+')[1]), data=temp, ax=ax)

    ax.set_ylim(0, 50)

    ax.set_xlim(0, 50)

    ax.set_title(score)

    

plt.tight_layout()

plt.show()
rate_res.to_csv('rate_res.csv', index=False)