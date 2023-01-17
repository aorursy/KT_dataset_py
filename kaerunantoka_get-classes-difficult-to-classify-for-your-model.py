import pandas as pd

import numpy as np

import os

import random

import torch

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



SEED = 1129



def seed_everything(seed=1129):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything(SEED)
# load competition data



class_map = pd.read_csv("../input/bengaliai-cv19/class_map.csv")

train = pd.read_csv("../input/bengaliai-cv19/train.csv")



y = train[["vowel_diacritic", "grapheme_root", "consonant_diacritic"]]
# get validation data which you used



train_idx, val_idx = train_test_split(train.index.tolist(), test_size=0.2, random_state=SEED, stratify=train["grapheme_root"])



y_val = y.values[val_idx].T

y_val.shape, y_val
# load your validation preds data



y_pred = np.load('/kaggle/input/bengali-valid-preds/bengali_valid_preds.npy')

y_pred.shape, y_pred
def get_mislabel_and_probs(y_pred_df, y_test_df, class_map, component_type, threshold):

    # Compute confusion matrix

    cnf_matrix = confusion_matrix(y_test_df, y_pred_df)

    

    # normalization

    cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    

    class_names = list(y_pred_df[0].unique())

    

    matrix_map = dict([(n, c) for n, c in enumerate(class_names)])

    res = [[dict([c for c in class_map[class_map['component_type']==component_type] \

           [['label', 'component']].values])[matrix_map[np.argmax(c)]], 

            matrix_map[np.argmax(c)],

            max(c)] for c in cm if max(c) < threshold]

    return res
# vowel_diacritic

y_test_df = pd.DataFrame(y_val[0]) # V

y_pred_df = pd.DataFrame(y_pred[0]) # V



get_mislabel_and_probs(y_pred_df, y_test_df, class_map, 'vowel_diacritic', 0.99)
# grapheme_root

y_test_df = pd.DataFrame(y_val[1]) # G

y_pred_df = pd.DataFrame(y_pred[1]) # G



get_mislabel_and_probs(y_pred_df, y_test_df, class_map, 'grapheme_root', 0.9)
# consonant_diacritic

y_test_df = pd.DataFrame(y_val[2]) # C

y_pred_df = pd.DataFrame(y_pred[2]) # C



get_mislabel_and_probs(y_pred_df, y_test_df, class_map, 'consonant_diacritic', 0.99)