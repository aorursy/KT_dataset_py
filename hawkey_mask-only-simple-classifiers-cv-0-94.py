import os

import gc

import sys

import cv2

import glob

import time

import signal

import shutil

import numpy as np

import pandas as pd

from collections import OrderedDict, Counter



from PIL import Image

import skimage.io

import cv2

import matplotlib.pyplot as plt



from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import StratifiedKFold



from tqdm.notebook import tqdm
DATA = "/kaggle/input/prostate-cancer-grade-assessment"

gls2isu = {"0+0":0,'negative':0,'3+3':1,'3+4':2,'4+3':3,'4+4':4,'3+5':4,'5+3':4,'4+5':5,'5+4':5,'5+5':5}
df_train = pd.read_csv(os.path.join(DATA, "train.csv"))

df_train = df_train[df_train.data_provider == 'radboud']
def extract_features(mask):

    counts = []

    for i in range(1,6):

        counts.append(np.count_nonzero(mask == i))

    percents = np.array(counts).astype(np.float32)

    percents /= percents.sum()

    return counts, percents



def quadratic_weighted_kappa(y_hat, y):

    return cohen_kappa_score(y_hat, y, weights='quadratic')



for label in range(1,6):

    df_train[f'percent_{label}'] = None

    df_train[f'count_{label}'] = None

    

for i in tqdm(range(len(df_train))):

    idx = df_train.iloc[i, 0]

    isup = df_train.iloc[i, 2]

    gleason = df_train.iloc[i, 3]

    

    mask_file = os.path.join(DATA, 'train_label_masks', f'{idx}_mask.tiff')

    if os.path.exists(mask_file):

        mask = skimage.io.MultiImage(mask_file)

        mask = np.array(mask[2]) # smallest resolution

        cnt, feat = extract_features(mask)

        for label in range(1,6):

            df_train[f'count_{label}'].iloc[i] = cnt[label-1]

            df_train[f'percent_{label}'].iloc[i] = feat[label-1]

    else:

        continue
df_train = df_train.replace(to_replace='None', value=np.nan).dropna()

df_train.reset_index(drop=True)
skf = StratifiedKFold(5, shuffle=True, random_state=42)

splits = list(skf.split(df_train, df_train.isup_grade))



#features = [f"percent_{label}" for label in range(1, 6)] 

features = [f"percent_{label}" for label in range(1, 6)] + [f"count_{label}" for label in range(1, 6)]

target = 'isup_grade'
# kNN

from sklearn.neighbors import KNeighborsClassifier



scores = []

for fold in range(5):

    train = df_train.iloc[splits[fold][0]]

    valid = df_train.iloc[splits[fold][1]]

    

    model = KNeighborsClassifier(n_neighbors=5)

    

    model.fit(train[features], train[target])

    

    preds = model.predict(valid[features])

    

    score = quadratic_weighted_kappa(preds, valid[target])

    scores.append(score)

    print(f"Fold = {fold}, QWK = {score:.4f}")

    

print(f"Mean = {np.mean(scores):.4f}")
# rfc

from sklearn.ensemble import RandomForestClassifier



scores = []

for fold in range(5):

    train = df_train.iloc[splits[fold][0]]

    valid = df_train.iloc[splits[fold][1]]

    

    model = RandomForestClassifier(random_state=42)

    

    model.fit(train[features], train[target])

    

    preds = model.predict(valid[features])

    

    score = quadratic_weighted_kappa(preds, valid[target])

    scores.append(score)

    print(f"Fold = {fold}, QWK = {score:.4f}")

    

print(f"Mean = {np.mean(scores):.4f}")
# lgb

import lightgbm as lgb



def QWK(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.rint(preds)

    score = quadratic_weighted_kappa(preds, labels)

    return ("QWK", score, True)



scores = []

for fold in range(5):

    train = df_train.iloc[splits[fold][0]]

    valid = df_train.iloc[splits[fold][1]]

    

    train_dataset = lgb.Dataset(train[features], train[target])

    valid_dataset = lgb.Dataset(valid[features], valid[target])

    

    params = {

                "objective": 'regression',

                "metric": 'rmse',

                "seed": 42,

                "learning_rate": 0.01,

                "boosting": "gbdt",

            }

        

    model = lgb.train(

                params=params,

                num_boost_round=1000,

                early_stopping_rounds=200,

                train_set=train_dataset,

                valid_sets=[train_dataset, valid_dataset],

                verbose_eval=100,

                feval=QWK,

            )

        

    

    preds = model.predict(valid[features], num_iteration=model.best_iteration)

    preds = np.rint(preds)

    

    score = quadratic_weighted_kappa(preds, valid[target])

    scores.append(score)

    

    print(f"Fold = {fold}, QWK = {score:.4f}")

    

print(f"Mean = {np.mean(scores):.4f}")