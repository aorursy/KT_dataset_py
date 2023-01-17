import os 
import random
from PIL import Image

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

input_path = '/kaggle/input/plant-pathology-2020-fgvc7'
train_df = pd.read_csv(os.path.join(input_path, 'train.csv'))
train_df.head()
sample_count = (train_df[['healthy', 'multiple_diseases', 'rust', 'scab']].sum())

ax = sns.barplot(x=sample_count.index, y=sample_count.values)
label_cols = train_df.columns[1:]

_, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(10, 3))
for ax, column in zip(axes, label_cols):
    train_df[column].value_counts().plot.bar(title=column, ax=ax)
plt.show()
img_dir = os.path.join(input_path, 'images')
path_list = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]
     
sizes = [Image.open(path).size for path in path_list]
pd.DataFrame(data = {'sizes': sizes}).sizes.value_counts()
train_df['image_id'] = train_df['image_id'] + '.jpg'
class_columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
class_label = train_df[class_columns].idxmax(axis=1)

train_df = pd.concat([train_df['image_id'], class_label], axis=1)
train_df.columns = ['image_id', 'label']
train_df.head(3)
n = 2
m = 2
files = random.sample(list(train_df['image_id']), n+m)
fig, axes = plt.subplots(n, m, figsize=(10,10))
k = 0
for axe in axes:
    for ax in axe:
        im = Image.open(os.path.join(input_path, 'images', files[k]))
        ax.imshow(np.asarray(im))
        ax.legend(train_df[train_df['image_id'] == files[k]])
        k += 1
def get_part(df, label):
    part_1 = train_df[train_df['label'] == label].copy()
    part_1['label'] = 1
    part_2 = train_df[train_df['label'] != label].copy()
    part_2['label'] = 0

    return pd.concat([part_1, part_2], axis=0).reset_index(drop=True)
scab_df = get_part(train_df, 'scab')
healthy_df = get_part(train_df, 'healthy')
rust_df = get_part(train_df, 'rust')
multi_dis_df = get_part(train_df, 'multiple_diseases')
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input 

from tqdm import tqdm
model = ResNet50(weights='imagenet', include_top=False, pooling='max')
def feature_extractor(im_name, model=model, img_dir=img_dir):
    im_path = os.path.join(img_dir, im_name)
    img = image.load_img(im_path, target_size=(224, 224, 3))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    feature = model.predict(img_data)
    return feature.flatten()

def get_features_labels(df):
    features = np.array([feature_extractor(img) for img in tqdm(df.image_id.values)])
    labels = np.array([label for label in tqdm(df.label.values)])
    return features, labels
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
scab_features, scab_labels = get_features_labels(scab_df)
multi_features, multi_labels = get_features_labels(multi_dis_df)
healthy_features, healthy_labels = get_features_labels(healthy_df)
rust_features, rust_labels = get_features_labels(rust_df)


SEED = 2
scab_X_tr, scab_X_v, scab_y_tr, scab_y_v = train_test_split(scab_features, scab_labels,
                                                            test_size=0.25, random_state=SEED)

multi_X_tr, multi_X_v, multi_y_tr, multi_y_v = train_test_split(multi_features, multi_labels,
                                                                test_size=0.25, random_state=SEED)

healthy_X_tr, healthy_X_v, healthy_y_tr, healthy_y_v = train_test_split(healthy_features, healthy_labels,
                                                            test_size=0.25, random_state=SEED)

rust_X_tr, rust_X_v, rust_y_tr, rust_y_v = train_test_split(rust_features, rust_labels,
                                                            test_size=0.25, random_state=SEED)
params = {'n_estimators': 800,
         'thread_count': -1,
         'depth': 6,
         'eval_metric': 'AUC',
         'verbose':100}

cbc_scab = CatBoostClassifier(**params)
cbc_multi = CatBoostClassifier(**params)
cbc_healthy = CatBoostClassifier(**params)
cbc_rust = CatBoostClassifier(**params)
cbc_scab.fit(scab_X_tr, scab_y_tr,
         eval_set=(scab_X_v, scab_y_v),
         use_best_model=True, plot=True)

cbc_multi.fit(multi_X_tr, multi_y_tr,
         eval_set=(multi_X_v, multi_y_v),
         use_best_model=True, plot=True)

cbc_healthy.fit(healthy_X_tr, healthy_y_tr,
         eval_set=(healthy_X_v, healthy_y_v),
         use_best_model=True, plot=True)

cbc_rust.fit(rust_X_tr, rust_y_tr,
         eval_set=(rust_X_v, rust_y_v),
         use_best_model=True, plot=True)
def classify(im_name):
    predict_list = []
    predict_list.append(im_name)
    
    im_name = im_name + '.jpg'
    im_features = feature_extractor(im_name)
    
    predict_list.append(cbc_healthy.predict(im_features))
    predict_list.append(cbc_multi.predict(im_features))
    predict_list.append(cbc_rust.predict(im_features))
    predict_list.append(cbc_scab.predict(im_features))
    return predict_list

def classify_prob(im_name):
    predict_list = []
    predict_list.append(im_name)
    
    im_name = im_name + '.jpg'
    im_features = feature_extractor(im_name)
    
    predict_list.append(cbc_healthy.predict_proba(im_features)[1])
    predict_list.append(cbc_multi.predict_proba(im_features)[1])
    predict_list.append(cbc_rust.predict_proba(im_features)[1])
    predict_list.append(cbc_scab.predict_proba(im_features)[1])
    return predict_list
test_df = pd.read_csv(os.path.join(input_path, 'test.csv'))
res_list = [classify_prob(i) for i in tqdm(test_df.image_id)]
# res_list = [classify(i) for i in tqdm(test_df.image_id)]
res_df = pd.DataFrame(res_list,
                      columns=['image_id', 'healthy', 'multiple_diseases',
                               'rust', 'scab'])
res_df.head()
res_df['norm'] = res_df.healthy + res_df.multiple_diseases + res_df.rust + res_df.scab
col_list = ['healthy', 'multiple_diseases', 'rust', 'scab']
for i in col_list:
    res_df[i] = res_df[i] / res_df['norm']

res_df.drop(columns=['norm'], inplace=True)
res_df.to_csv('submission_7_1.csv', index=False)