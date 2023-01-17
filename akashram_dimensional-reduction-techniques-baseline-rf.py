import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

import gc

import json

import math

import cv2

from PIL import Image

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

#matplotlib.interactive(False)

import scipy

from tqdm import tqdm

%matplotlib inline



from keras.preprocessing import image

from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

test_df = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
def preprocess_image(image_path, desired_size=128):

    im = Image.open(image_path)

    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)

    return im
def convert_to_array(df, size=128):

    N = df.shape[0]

    x_train = np.empty((N, size, size, 3), dtype=np.uint8)

    for i, image_id in enumerate(tqdm(df['image_name'])):

        x_train[i, :, :, :] = preprocess_image(

            f'../input/siim-isic-melanoma-classification/jpeg/train/{image_id}.jpg'

        )

    return x_train
#x_train = convert_to_array(train_df)

x_train = np.load('../input/x-train-128npy/x_train_128.npy')
x_t = x_train[:2000]

x_t.shape

#train = x_train.reshape((x_train.shape[0], 128*128*3))
image = []

for i in range(0,2000):

    img = x_t[i].flatten()

    image.append(img)

image = np.array(image)
feat_cols = ['pixel'+str(i) for i in range(image.shape[1])]

df = pd.DataFrame(image,columns=feat_cols)

df['label'] = train_df['benign_malignant'][:2000]

df['label1'] = train_df['target'][:2000]
from sklearn.decomposition import FactorAnalysis



FA = FactorAnalysis(n_components=2).fit_transform(df[feat_cols].values)



# Here, n_components will decide the number of factors in the transformed data. After transforming the data, it’s time to visualize the results:



fa_data = np.vstack((FA.T, df['label1'])).T

fa_df = pd.DataFrame(fa_data, columns=['1st Component', '2nd Component', 'Label'])

sns.FacetGrid(fa_df, hue="Label", size=10).map(plt.scatter, "1st Component", "2nd Component").add_legend()

plt.show()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca_result = pca.fit_transform(df[feat_cols].values)
plt.figure(figsize=(12.5, 8))

plt.plot(range(2), pca.explained_variance_ratio_)

plt.plot(range(2), np.cumsum(pca.explained_variance_ratio_))

plt.title("Component-wise and Cumulative Explained Variance")
from sklearn.decomposition import PCA 

pca = PCA(n_components=2, random_state=42).fit_transform(df[feat_cols].values)

pca_data = np.vstack((pca.T, df['label1'])).T

pca_df = pd.DataFrame(pca_data, columns=['1st Component', '2nd Component', 'Label'])

sns.FacetGrid(pca_df, hue="Label", size=10).map(plt.scatter, "1st Component", "2nd Component").add_legend()

plt.show()
from sklearn.decomposition import TruncatedSVD 

svd = TruncatedSVD(n_components=2, random_state=42).fit_transform(df[feat_cols].values)

svd_data = np.vstack((svd.T, df['label1'])).T

svd_df = pd.DataFrame(svd_data, columns=['1st Component', '2nd Component', 'Label'])

sns.FacetGrid(svd_df, hue="Label", size=10).map(plt.scatter, "1st Component", "2nd Component").add_legend()

plt.show()
from sklearn.manifold import TSNE 

tsne = TSNE(n_components=2).fit_transform(df[feat_cols].values)

ts_data = np.vstack((tsne.T, df['label1'])).T

ts_df = pd.DataFrame(ts_data, columns=['1st Component', '2nd Component', 'Label'])

sns.FacetGrid(ts_df, hue="Label", size=10).map(plt.scatter, "1st Component", "2nd Component").add_legend()

plt.show()
import umap

umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2).fit_transform(df[feat_cols].values)



#Here,

#n_neighbors determines the number of neighboring points used

#min_dist controls how tightly embedding is allowed. Larger values ensure embedded points are more evenly distributed



umap_data = np.vstack((umap_data.T, df['label1'])).T

umap_df = pd.DataFrame(umap_data, columns=['1st Component', '2nd Component', 'Label'])

sns.FacetGrid(umap_df, hue="Label", size=10).map(plt.scatter, "1st Component", "2nd Component").add_legend()

plt.show()
def convert_to_tarray(df, size=128):

    N = df.shape[0]

    x_test = np.empty((N, size, size, 3), dtype=np.uint8)

    for i, image_id in enumerate(tqdm(df['image_name'])):

        x_test[i, :, :, :] = preprocess_image(

            f'../input/siim-isic-melanoma-classification/jpeg/test/{image_id}.jpg'

        )

    return x_test



x_test = convert_to_tarray(test_df)
x_train = x_train.reshape((x_train.shape[0], 128*128*3))

#x_test = x_test.reshape((x_test.shape[0], 128*128*3))

y = train_df.target.values
train_oof = np.zeros((x_train.shape[0], ))

test_preds = 0
n_splits = 3



kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)



for ij, (train_index, val_index) in enumerate(kf.split(x_train)):

    

    print("Fitting fold", ij+1)

    train_features = x_train[train_index]

    train_target = y[train_index]

    

    val_features = x_train[val_index]

    val_target = y[val_index]

        

    model = RandomForestClassifier(max_depth=2, random_state=0)

    model.fit(train_features, train_target)

    

    val_pred = model.predict_proba(val_features)[:,1]

    

    train_oof[val_index] = val_pred

    

    print("Fold AUC:", roc_auc_score(val_target, val_pred))

    test_preds += model.predict_proba(x_test)[:,1]/n_splits

    

    del train_features, train_target, val_features, val_target

    gc.collect()

    

print(roc_auc_score(y, train_oof))
sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')



sample_submission['target'] = test_preds



sample_submission.to_csv('submission_RF_01.csv', index=False)



sample_submission['target'].max()



sample_submission['target'].min()