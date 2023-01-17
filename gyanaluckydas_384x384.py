import pandas as pd

import numpy as np

from glob import glob

import cv2

from skimage import io

from tqdm import tqdm

import seaborn as sns
!mkdir '384x384-dataset-melanoma'

!mkdir '384x384-test'
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
image.shape
print('[ALL]:', df_meta.shape[0])

print('[∩ isic2020]:', len(set(df_meta['image_id'].values).intersection(df_train['image_name'].values)))

print('[∩ isic2019]:', len(set(df_meta['image_id'].values).intersection(df_gt['image'].values)))

print('[∩ slatmd]:', len(set(df_meta['image_id'].values).intersection(image_ids)))
NEED_IMAGE_SAVE = True
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

        image = cv2.resize(image, (384, 384), cv2.INTER_AREA)

        cv2.imwrite(f'./384x384-dataset-melanoma/{image_id}.jpg', image)



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

        image = cv2.resize(image, (384, 384), cv2.INTER_AREA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f'./384x384-dataset-melanoma/{image_id}.jpg', image)



    

dataset = pd.DataFrame(dataset).set_index('image_id')    
dataset.head()
df_duplicates = pd.read_csv('../input/melanoma-merged-external-data-512x512-jpeg/duplicates_13062020.csv', index_col='image_ids')
def get_value(duplicate_data, row, field):

    if row[field] == 1 or duplicate_data.shape[0] <= 2:

        return duplicate_data.iloc[0][field]

    if 'ISIC20' in duplicate_data.source.values:

        duplicate_data = duplicate_data[duplicate_data.source == 'ISIC20']

    return sorted(duplicate_data[field].value_counts().items(), key=lambda x: -x[1])[0][0]
cleaned_duplicates = {

    'image_id': [],

    'patient_id': [],

    'target': [],

    'source': [],

    'sex': [],

    'age_approx': [],

    'anatom_site_general_challenge': [],

}

drop_image_ids = []

for image_ids, row in df_duplicates.iterrows():

    image_ids = image_ids.split('.')

    drop_image_ids.extend(image_ids)

    duplicate_data = dataset.loc[image_ids].sort_values('source', ascending=False)

    for field in [    

        'patient_id',

        'target',

        'source',

        'sex',

        'age_approx',

        'anatom_site_general_challenge',

    ]:

        cleaned_duplicates[field].append(get_value(duplicate_data, row, field))

    cleaned_duplicates['image_id'].append(duplicate_data.index.values[0])



cleaned_duplicates = pd.DataFrame(cleaned_duplicates).set_index('image_id')

cleaned_duplicates.head()
dataset = dataset.drop(drop_image_ids)

dataset = dataset.append(cleaned_duplicates)
dataset.to_csv('marking.csv')
dataset['source'].hist();
print(dataset['target'].value_counts())

dataset['target'].hist();
dataset['sex'].fillna('unknown').hist();
dataset['age_approx'].hist(bins=50);
dataset['anatom_site_general_challenge'].fillna('unknown').value_counts()
import numpy as np

import random

import pandas as pd

from collections import Counter, defaultdict



def stratified_group_k_fold(X, y, groups, k, seed=None):

    """ https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation """

    labels_num = np.max(y) + 1

    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))

    y_distr = Counter()

    for label, g in zip(y, groups):

        y_counts_per_group[g][label] += 1

        y_distr[label] += 1



    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))

    groups_per_fold = defaultdict(set)



    def eval_y_counts_per_fold(y_counts, fold):

        y_counts_per_fold[fold] += y_counts

        std_per_label = []

        for label in range(labels_num):

            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])

            std_per_label.append(label_std)

        y_counts_per_fold[fold] -= y_counts

        return np.mean(std_per_label)

    

    groups_and_y_counts = list(y_counts_per_group.items())

    random.Random(seed).shuffle(groups_and_y_counts)



    for g, y_counts in tqdm(sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])), total=len(groups_and_y_counts)):

        best_fold = None

        min_eval = None

        for i in range(k):

            fold_eval = eval_y_counts_per_fold(y_counts, i)

            if min_eval is None or fold_eval < min_eval:

                min_eval = fold_eval

                best_fold = i

        y_counts_per_fold[best_fold] += y_counts

        groups_per_fold[best_fold].add(g)



    all_groups = set(groups)

    for i in range(k):

        train_groups = all_groups - groups_per_fold[i]

        test_groups = groups_per_fold[i]



        train_indices = [i for i, g in enumerate(groups) if g in train_groups]

        test_indices = [i for i, g in enumerate(groups) if g in test_groups]



        yield train_indices, test_indices
%%time



df_folds = pd.read_csv('marking.csv')

df_folds['patient_id'] = df_folds['patient_id'].fillna(df_folds['image_id'])

df_folds['sex'] = df_folds['sex'].fillna('unknown')

df_folds['anatom_site_general_challenge'] = df_folds['anatom_site_general_challenge'].fillna('unknown')

df_folds['age_approx'] = df_folds['age_approx'].fillna(round(df_folds['age_approx'].mean()))



patient_id_2_count = df_folds[['patient_id', 'image_id']].groupby('patient_id').count()['image_id'].to_dict()



df_folds = df_folds.set_index('image_id')



def get_stratify_group(row):

    stratify_group = row['sex']

#     stratify_group += f'_{row["anatom_site_general_challenge"]}'

    stratify_group += f'_{row["source"]}'

    stratify_group += f'_{row["target"]}'

    patient_id_count = patient_id_2_count[row["patient_id"]]

    if patient_id_count > 80:

        stratify_group += f'_80'

    elif patient_id_count > 60:

        stratify_group += f'_60'

    elif patient_id_count > 50:

        stratify_group += f'_50'

    elif patient_id_count > 30:

        stratify_group += f'_30'

    elif patient_id_count > 20:

        stratify_group += f'_20'

    elif patient_id_count > 10:

        stratify_group += f'_10'

    else:

        stratify_group += f'_0'

    return stratify_group



df_folds['stratify_group'] = df_folds.apply(get_stratify_group, axis=1)

df_folds['stratify_group'] = df_folds['stratify_group'].astype('category').cat.codes
%%time



df_folds.loc[:, 'fold'] = 0



skf = stratified_group_k_fold(X=df_folds.index, y=df_folds['stratify_group'], groups=df_folds['patient_id'], k=5, seed=42)



for fold_number, (train_index, val_index) in enumerate(skf):

    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
set(df_folds[df_folds['fold'] == 0]['patient_id'].values).intersection(df_folds[df_folds['fold'] == 1]['patient_id'].values)
df_folds[df_folds['fold'] == 0]['target'].hist();
df_folds[df_folds['fold'] == 1]['target'].hist();
df_folds.to_csv('folds_13062020.csv')
# test isic2020

df_test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv', index_col='image_name')

for image_id, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):   

    if NEED_IMAGE_SAVE:

        image = cv2.imread(f'../input/siim-isic-melanoma-classification/jpeg/test/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (384, 384), cv2.INTER_AREA)

        cv2.imwrite(f'./384x384-test/{image_id}.jpg', image)
df_folds = pd.read_csv('../input/melanoma-merged-external-data-512x512-jpeg/folds_13062020.csv')

df_folds.head()
image = cv2.imread(f'../input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/ISIC_0074268.jpg', cv2.IMREAD_COLOR)

io.imshow(image);
image = cv2.imread(f'../input/melanoma-merged-external-data-512x512-jpeg/512x512-test/512x512-test/ISIC_0089356.jpg', cv2.IMREAD_COLOR)

io.imshow(image);