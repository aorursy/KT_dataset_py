!pip install autokeras

!pip install natsort
import autokeras as ak

import numpy as np 

import pandas as pd 

from glob import glob

from skimage.io import imread

import skimage.io as sio

import os

from natsort import natsorted

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from skimage.transform import resize, rotate

import warnings; warnings.filterwarnings("ignore")
# train test directories

root_dir = "../input"

train_dir = root_dir + "/train/train/"

test_dir  = root_dir + "/test/test/"

csv_path  = root_dir + "/train.csv"

sub_path  = root_dir + "sample_submission.csv"



# loading images

df   = pd.read_csv(csv_path)

x    = np.array([ imread(train_dir+p)/255 for p in df.id.values])

y    = df.has_cactus.values
# splitting training dataset into train/validation

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20,stratify=y)
# helper functions here

def display_images(imgs,y=None, y_pred=None):

    n_images = imgs.shape[0]

    n_gridx  = 5

    n_gridy  = n_images//n_gridx

#     n_grid   = int(np.sqrt(n_images))

    k = 1

    plt.figure(figsize=(10,6),frameon=False)

    for i in range(n_gridy):

        for j in range(n_gridx):

            plt.subplot(n_gridy, n_gridx, k)

            plt.imshow(imgs[k-1])

            plt.axis("off")

            if (y is not None) and (y_pred is not None):

                plt.title("y=%d | pred=%0.1f"%(y[k-1],y_pred[k-1]))

            elif y is not None:

                plt.title("y=%d"%y[k-1])

            k+=1

    plt.tight_layout()

    plt.show()





def getProb(model, x):

    xprocessed = model.preprocess(x)

    loader = model.data_transformer.transform_test(xprocessed)

    probs  = model.cnn.predict(loader)

    num    = np.exp(probs[:,1])

    denom  = num + np.exp(probs[:,0])

    probs  = num / denom 

    return probs
n_samples  = 20

idx_sample = np.random.randint(0,len(x_train),n_samples)

display_images(x_train[idx_sample], y_train[idx_sample])
runFor = 5 # time in hours

model = ak.ImageClassifier(verbose=True, augment=True )

model.fit(x_train, y_train, time_limit=4*60*60)
# model.final_fit(x_train, y_train, x_val, y_val, retrain=False)

y_pred = model.predict(x_train)

y_prob = getProb(model, x_train)

print("training   accuracy  = ", accuracy_score(y_train, y_pred))

print("training   recall    = ", recall_score(y_train, y_pred))

print("training   precision = ", precision_score(y_train, y_pred))

print("training   auc score = ", roc_auc_score(y_train, y_prob))

print("training   f1 score  = ", f1_score(y_train, y_pred))

y_pred = model.predict(x_val)

y_prob = getProb(model, x_val)

print("validation accuracy  = ", accuracy_score(y_val, y_pred))

print("validation recall    = ", recall_score(y_val, y_pred))

print("validation precision = ", precision_score(y_val, y_pred))

print("validation auc score = ",roc_auc_score(y_val, y_prob))

print("validation f1 score  = ", f1_score(y_val, y_pred))
df_test = pd.read_csv('../input/sample_submission.csv')

x_test  = np.array([ imread(test_dir+p)/255 for p in df_test.id.values])

x_test  = np.array(x_test)



# test prediction

y_prob_test = getProb(model, x_test)



df_test['has_cactus'] = y_prob_test

df_test.to_csv('cactus_net_submission.csv', index=False)