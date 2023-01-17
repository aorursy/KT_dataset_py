import pandas as pd
import numpy as np
from glob import glob
import cv2
from skimage import io
from tqdm import tqdm
import seaborn as sns
import os
!mkdir '224x224-dataset-melanoma'
!mkdir '224x224-test'
df_train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
df_gt = pd.read_csv('../input/isic-2019/ISIC_2019_Training_GroundTruth.csv')
image_id = df_gt.iloc[25]['image']
image = cv2.imread(f'../input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/{image_id}.jpg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
io.imshow(image);
print('[ALL]:', df_gt.shape[0])
print('[∩ isic2020]:', len(set(df_train['image_name'].values).intersection(df_gt['image'].values)))
paths = glob('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/*/*/*.jpg')
print(len(paths))
image = cv2.imread(paths[2], cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
io.imshow(image);
image_ids = [path.split('/')[-1][:-4] for path in paths]
print('[ALL]:', len(image_ids))
print('[∩ isic2020]:', len(set(image_ids).intersection(df_train['image_name'].values)))
print('[∩ isic2019]:', len(set(image_ids).intersection(df_gt['image'].values)))
df_meta = pd.read_csv('../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')
image_id = df_meta.iloc[777]['image_id']
image = cv2.imread(f'../input/skin-cancer-mnist-ham10000/HAM10000_images_part_1/{image_id}.jpg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
io.imshow(image);
print('[ALL]:', df_meta.shape[0])
print('[∩ isic2020]:', len(set(df_meta['image_id'].values).intersection(df_train['image_name'].values)))
print('[∩ isic2019]:', len(set(df_meta['image_id'].values).intersection(df_gt['image'].values)))
print('[∩ slatmd]:', len(set(df_meta['image_id'].values).intersection(image_ids)))
NEED_IMAGE_SAVE = True
if not os.path.exists('./224x224-dataset-melanoma/melanoma'):
    os.makedirs('./224x224-dataset-melanoma/melanoma')
    
if not os.path.exists('./224x224-dataset-melanoma/other'):
    os.makedirs('./224x224-dataset-melanoma/other')
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
        image = cv2.resize(image, (224,224), cv2.INTER_AREA)
        dest = f'./224x224-dataset-melanoma/other/{image_id}.jpg'
        if row['target']==1:
            dest = f'./224x224-dataset-melanoma/melanoma/{image_id}.jpg'
        cv2.imwrite(dest, image)

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
        image = cv2.resize(image, (224,224), cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dest = f'./224x224-dataset-melanoma/other/{image_id}.jpg'
        if int(df_gt.loc[image_id]['MEL'])==1:
            dest = f'./224x224-dataset-melanoma/melanoma/{image_id}.jpg'
        cv2.imwrite(dest, image)


# skin-lesion-analysis-toward-melanoma-detection
paths = glob('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/*/*/*.jpg')
for path in tqdm(paths, total=len(paths)):
    diagnosis, image_id = path.split('/')[-2:]
    image_id = image_id[:-4]
    
    if image_id in dataset['image_id']:
        continue
    
    target = int(diagnosis == 'melanoma')
    dataset['patient_id'].append(np.nan)
    dataset['image_id'].append(image_id)
    dataset['target'].append(target)
    dataset['source'].append('SLATMD')
    dataset['sex'].append(np.nan)
    dataset['age_approx'].append(np.nan)
    dataset['anatom_site_general_challenge'].append(np.nan)
    
    if NEED_IMAGE_SAVE:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224), cv2.INTER_AREA)
        dest = f'./224x224-dataset-melanoma/other/{image_id}.jpg'
        if target==1:
            dest = f'./224x224-dataset-melanoma/melanoma/{image_id}.jpg'
        cv2.imwrite(dest, image)
    
dataset = pd.DataFrame(dataset)    
dataset.head()
dataset.to_csv('merged_data.csv', index=False)
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

df_folds = pd.read_csv(f'merged_data.csv')
df_folds['patient_id'] = df_folds['patient_id'].fillna(df_folds['image_id'])
df_folds['sex'] = df_folds['sex'].fillna('unknown')
df_folds['anatom_site_general_challenge'] = df_folds['anatom_site_general_challenge'].fillna('unknown')
df_folds['age_approx'] = df_folds['age_approx'].fillna(round(df_folds['age_approx'].mean()))
df_folds = df_folds.set_index('image_id')

def get_stratify_group(row):
    stratify_group = row['sex']
    stratify_group += f'_{row["anatom_site_general_challenge"]}'
    stratify_group += f'_{row["source"]}'
    stratify_group += f'_{row["target"]}'
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
df_folds.to_csv('folds.csv')
# test isic2020
df_test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv', index_col='image_name')

for image_id, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):   
    if NEED_IMAGE_SAVE:
        image = cv2.imread(f'../input/siim-isic-melanoma-classification/jpeg/test/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224), cv2.INTER_AREA)
        cv2.imwrite(f'kaggle/working/224x224-test/{image_id}.jpg', image)
import zipfile
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file == 'all_data.zip':
                ziph.write(os.path.join(root, file))
zipf = zipfile.ZipFile('all_data.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('/kaggle/working', zipf)
zipf.close()
