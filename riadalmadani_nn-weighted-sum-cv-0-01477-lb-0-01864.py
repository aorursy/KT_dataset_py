import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np

import random

import pandas as pd

import matplotlib.pyplot as plt

import os

import copy

import seaborn as sns



from sklearn import preprocessing

from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import warnings

warnings.filterwarnings('ignore')
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')



test_features = pd.read_csv('../input/lish-moa/test_features.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
GENES = [col for col in train_features.columns if col.startswith('g-')]

CELLS = [col for col in train_features.columns if col.startswith('c-')]
def seed_everything(seed=321):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(seed=321)
# GENES

n_comp = 29



data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

data2 = (PCA(n_components=n_comp, random_state=321).fit_transform(data[GENES]))

train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]

train_features = pd.concat((train_features, train2), axis=1)

test_features = pd.concat((test_features, test2), axis=1)



#CELLS

n_comp = 5



data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

data2 = (PCA(n_components=n_comp, random_state=321).fit_transform(data[CELLS]))

train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]

train_features = pd.concat((train_features, train2), axis=1)

test_features = pd.concat((test_features, test2), axis=1)
from sklearn.feature_selection import VarianceThreshold





var_thresh = VarianceThreshold(threshold=0.4)

data = train_features.append(test_features)

data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])



train_features_transformed = data_transformed[ : train_features.shape[0]]

test_features_transformed = data_transformed[-test_features.shape[0] : ]





train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\

                              columns=['sig_id','cp_type','cp_time','cp_dose'])



train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)





test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\

                             columns=['sig_id','cp_type','cp_time','cp_dose'])



test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)



train_features
train = train_features.merge(train_targets_scored, on='sig_id')

train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)

test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)



target = train[train_targets_scored.columns]



train = train.drop('cp_type', axis=1)

test = test.drop('cp_type', axis=1)

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
folds = train.copy()



mskf = MultilabelStratifiedKFold(n_splits=5)



for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):

    folds.loc[v_idx, 'kfold'] = int(f)



folds['kfold'] = folds['kfold'].astype(int)
class MoADataset:

    def __init__(self, features, targets):

        self.features = features

        self.targets = targets

        

    def __len__(self):

        return (self.features.shape[0])

    

    def __getitem__(self, idx):

        dct = {

            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),

            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            

        }

        return dct

    

class TestDataset:

    def __init__(self, features):

        self.features = features

        

    def __len__(self):

        return (self.features.shape[0])

    

    def __getitem__(self, idx):

        dct = {

            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)

        }

        return dct
def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):

    model.train()

    final_loss = 0

    

    for data in dataloader:

        optimizer.zero_grad()

        inputs, targets = data['x'].to(device), data['y'].to(device)

        #print(inputs.shape)

        outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        loss.backward()

        optimizer.step()

        scheduler.step()

        

        final_loss += loss.item()

        

    final_loss /= len(dataloader)

    

    return final_loss





def valid_fn(model, loss_fn, dataloader, device):

    model.eval()

    final_loss = 0

    valid_preds = []

    

    for data in dataloader:

        inputs, targets = data['x'].to(device), data['y'].to(device)

        outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        

        final_loss += loss.item()

        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

        

    final_loss /= len(dataloader)

    valid_preds = np.concatenate(valid_preds)

    

    return final_loss, valid_preds



def inference_fn(model, dataloader, device):

    model.eval()

    preds = []

    

    for data in dataloader:

        inputs = data['x'].to(device)



        with torch.no_grad():

            outputs = model(inputs)

        

        preds.append(outputs.sigmoid().detach().cpu().numpy())

        

    preds = np.concatenate(preds)

    

    return preds

   
class Model(nn.Module):

    def __init__(self, num_features, num_targets, hidden_size):

        super(Model, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(num_features)

        self.dropout1 = nn.Dropout(0.2)

        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

        self.dropout2 = nn.Dropout(0.3)

        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)

        self.dropout3 = nn.Dropout(0.25)

        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    

    def forward(self, x):

        x = self.batch_norm1(x)

        x = self.dropout1(x)

        x = F.relu(self.dense1(x))

        

        x = self.batch_norm2(x)

        x = self.dropout2(x)

        x = F.relu(self.dense2(x))

        

        x = self.batch_norm3(x)

        x = self.dropout3(x)

        x = self.dense3(x)

        

        return x
def process_data(data):

    

    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])

#     data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})

#     data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})



# --------------------- Normalize ---------------------

#     for col in GENES:

#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))

    

#     for col in CELLS:

#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))

    

#--------------------- Removing Skewness ---------------------

#     for col in GENES + CELLS:

#         if(abs(data[col].skew()) > 0.75):

            

#             if(data[col].skew() < 0): # neg-skewness

#                 data[col] = data[col].max() - data[col] + 1

#                 data[col] = np.sqrt(data[col])

            

#             else:

#                 data[col] = np.sqrt(data[col])

    

    return data
feature_cols = [c for c in process_data(folds).columns if c not in target_cols]

feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]

len(feature_cols)
# HyperParameters



DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 25

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

NFOLDS = 5

EARLY_STOPPING_STEPS = 25

EARLY_STOP = True



num_features=len(feature_cols)

num_targets=len(target_cols)

hidden_size=2048
def run_training(fold, seed):

    

    seed_everything(seed)

    

    train = process_data(folds)

    test_ = process_data(test)

    

    trn_idx = train[train['kfold'] != fold].index

    val_idx = train[train['kfold'] == fold].index

    

    train_df = train[train['kfold'] != fold].reset_index(drop=True)

    valid_df = train[train['kfold'] == fold].reset_index(drop=True)

    

    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values

    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values

    

    train_dataset = MoADataset(x_train, y_train)

    valid_dataset = MoADataset(x_valid, y_valid)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    

    model = Model(

        num_features=num_features,

        num_targets=num_targets,

        hidden_size=hidden_size,

    )

    

    model.to(DEVICE)

    

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 

                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    

    loss_fn = nn.BCEWithLogitsLoss()

    

    early_stopping_steps = EARLY_STOPPING_STEPS

    early_step = 0

    

    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))

    best_loss = np.inf

    

    for epoch in range(EPOCHS):

        

        train_loss = train_fn(model, optimizer,scheduler, loss_fn, trainloader, DEVICE)

        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")

        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)

        print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")

        

        if valid_loss < best_loss:

            

            best_loss = valid_loss

            oof[val_idx] = valid_preds

            torch.save(model.state_dict(), f"FOLD{fold}_.pth")

        

        elif(EARLY_STOP == True):

            

            early_step += 1

            if (early_step >= early_stopping_steps):

                break

            

    

    #--------------------- PREDICTION---------------------

    x_test = test_[feature_cols].values

    testdataset = TestDataset(x_test)

    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    

    model = Model(

        num_features=num_features,

        num_targets=num_targets,

        hidden_size=hidden_size,

    )

    

    model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))

    model.to(DEVICE)

    

    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))

    predictions = inference_fn(model, testloader, DEVICE)

    

    return oof, predictions
def run_k_fold(NFOLDS, seed):

    oof = np.zeros((len(train), len(target_cols)))

    predictions = np.zeros((len(test), len(target_cols)))

    

    for fold in range(NFOLDS):

        oof_, pred_ = run_training(fold, seed)

        

        predictions += pred_ / NFOLDS

        oof += oof_

        

    return oof, predictions
# Averaging on multiple SEEDS



SEED = [456]

oof = np.zeros((len(train), len(target_cols)))

predictions = np.zeros((len(test), len(target_cols)))



for seed in SEED:

    

    oof_, predictions_ = run_k_fold(NFOLDS, seed)

    oof += oof_ / len(SEED)

    predictions += predictions_ / len(SEED)



train[target_cols] = oof

test[target_cols] = predictions
valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)





y_true = train_targets_scored[target_cols].values

y_pred = valid_results[target_cols].values



score = 0

for i in range(len(target_cols)):

    score_ = log_loss(y_true[:, i], y_pred[:, i])

    score += score_ / target.shape[1]

    

print("CV log_loss: ", score)
sub= sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
import os

import gc

import pickle

import datetime

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from sklearn.decomposition import PCA

from tqdm.notebook import tqdm

from time import time
import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import tensorflow_addons as tfa

from sklearn.model_selection import KFold
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')



ss_svc = pd.read_csv('../input/lish-moa/sample_submission.csv')



ss_lr = ss_svc.copy()

ss_rf = ss_svc.copy()

ss    = ss_svc.copy()

cols = [c for c in ss_svc.columns.values if c != 'sig_id']
def preprocess(df):

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del df['sig_id']

    return df



def log_loss_metric(y_true, y_pred):

    metrics = []

    for _target in train_targets.columns:

        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels = [0,1]))

    return np.mean(metrics)



train = preprocess(train_features)

test = preprocess(test_features)



del train_targets['sig_id']
def create_model1(num_columns):

    model = tf.keras.Sequential([

    tf.keras.layers.Input(num_columns),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(512, activation="elu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(1024, activation="elu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(512, activation="elu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="elu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),   

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation="sigmoid"))

    ])

    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),

                  loss='binary_crossentropy', 

                  )

    return model
def create_model2(num_columns):

    model = tf.keras.Sequential([

    tf.keras.layers.Input(num_columns),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.2),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.2),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(1500, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.2),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(1048, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(500, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.2),   

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation="sigmoid"))

    ])

    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),

                  loss='binary_crossentropy', 

                  )

    return model
top_feats = [  1,   2,   3,   4,   5,   6,   7,   9,  11,  14,  15,  16,  17,

        18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  29,  30,  31,

        32,  33,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  46,

        47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  58,  59,  60,

        61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,

        74,  75,  76,  78,  79,  80,  81,  82,  83,  84,  86,  87,  88,

        89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,

       102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,

       115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128,

       129, 130, 131, 132, 133, 136, 137, 138, 139, 140, 141, 142, 143,

       144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157,

       158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,

       171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,

       184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197,

       198, 199, 200, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212,

       213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226,

       227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,

       240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,

       254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,

       267, 268, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,

       281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 294,

       295, 296, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,

       310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,

       324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336,

       337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,

       350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362,

       363, 364, 365, 366, 367, 368, 369, 370, 371, 374, 375, 376, 377,

       378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 391,

       392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404,

       405, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418,

       419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,

       432, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446,

       447, 448, 449, 450, 453, 454, 456, 457, 458, 459, 460, 461, 462,

       463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475,

       476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489,

       490, 491, 492, 493, 494, 495, 496, 498, 500, 501, 502, 503, 505,

       506, 507, 509, 510, 511, 512, 513, 514, 515, 518, 519, 520, 521,

       522, 523, 524, 525, 526, 527, 528, 530, 531, 532, 534, 535, 536,

       538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 549, 550, 551,

       552, 554, 557, 559, 560, 561, 562, 565, 566, 567, 568, 569, 570,

       571, 572, 573, 574, 575, 577, 578, 580, 581, 582, 583, 584, 585,

       586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 599,

       600, 601, 602, 606, 607, 608, 609, 611, 612, 613, 615, 616, 617,

       618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630,

       631, 632, 633, 634, 635, 636, 637, 638, 639, 641, 642, 643, 644,

       645, 646, 647, 648, 649, 650, 651, 652, 654, 655, 656, 658, 659,

       660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,

       673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,

       686, 687, 688, 689, 691, 692, 693, 694, 695, 696, 697, 699, 700,

       701, 702, 704, 705, 707, 708, 709, 710, 711, 713, 714, 716, 717,

       718, 720, 721, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732,

       733, 734, 735, 737, 738, 739, 740, 742, 743, 744, 745, 746, 747,

       748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 759, 760, 761,

       762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774,

       775, 776, 777, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788,

       789, 790, 792, 793, 794, 795, 796, 797, 798, 800, 801, 802, 803,

       804, 805, 806, 808, 809, 811, 813, 814, 815, 816, 817, 818, 819,

       821, 822, 823, 825, 826, 827, 828, 829, 830, 831, 832, 834, 835,

       837, 838, 839, 840, 841, 842, 845, 846, 847, 848, 850, 851, 852,

       854, 855, 856, 858, 859, 860, 861, 862, 864, 866, 867, 868, 869,

       870, 871, 872, 873, 874]

print(len(top_feats))
N_STARTS = 2

tf.random.set_seed(42)





ss_NN = ss_svc.copy()

res_NN = train_targets.copy()



ss_NN.loc[:, train_targets.columns] = 0

res_NN.loc[:, train_targets.columns] = 0

for seed in range(N_STARTS):

    for n, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=4, random_state=seed, shuffle=True).split(train_targets, train_targets)):

        print(f'Fold {n}')

        

        if seed%2 == 0:

            model = create_model1(len(top_feats))

        else :

            model = create_model2(len(top_feats))



            

        checkpoint_path = f'repeat:{seed}_Fold:{n}.hdf5'

        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')



        cb_checkpt = ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True,

                                     save_weights_only = True, mode = 'min')

        model.fit(train.values[tr][:, top_feats],

                  train_targets.values[tr],

                  validation_data=(train.values[te][:, top_feats], train_targets.values[te]),

                  epochs=35, batch_size=128,

                  callbacks=[reduce_lr_loss, cb_checkpt], verbose=2

                 )

        

        model.load_weights(checkpoint_path)

        test_predict = model.predict(test.values[:, top_feats])

        val_predict = model.predict(train.values[te][:, top_feats])

        

        ss_NN.loc[:, train_targets.columns] += test_predict

        res_NN.loc[te, train_targets.columns] += val_predict

        print('')

    

ss_NN.loc[:, train_targets.columns] /= ((n+1) * N_STARTS)

res_NN.loc[:, train_targets.columns] /= N_STARTS
print(f'NN OOF Metric: {log_loss_metric(train_targets, res_NN)}')

res_NN.loc[train['cp_type'] == 1, train_targets.columns] = 0

ss_NN.loc[test['cp_type'] == 1, train_targets.columns] = 0

print(f'NN OOF Metric with postprocessing: {log_loss_metric(train_targets, res_NN)}')
old_list = [4, 5, 17, 20, 21, 23, 25, 26, 28, 29, 30, 31, 33, 40, 41, 42, 44, 49, 52, 55, 56, 60, 64, 66, 70, 72, 74, 76, 77, 79, 83, 84, 87, 91, 92, 93, 97, 98, 100, 106, 107, 108, 115, 124, 125, 128, 131, 132, 134, 135, 139, 140, 143, 144, 147, 148, 152, 155, 160, 162, 165, 173, 174, 176, 177, 179, 181, 183, 186, 190, 195, 196, 198, 199, 203, 205]
avg_list = []

new_list = []

for tar in range(206):

        score1 = log_loss(train_targets.loc[:, train_targets.columns[tar]], valid_results.loc[:, train_targets.columns[tar]])

        score2 = log_loss(train_targets.loc[:, train_targets.columns[tar]], res_NN.loc[:, train_targets.columns[tar]])

        if score2 >= score1:

            ss_NN.loc[:, train_targets.columns[tar]] = sub.loc[:, train_targets.columns[tar]]

            res_NN.loc[:, train_targets.columns[tar]] = valid_results.loc[:, train_targets.columns[tar]] 

        else :

            if tar in old_list:

                avg_list.append(tar)

            else:

                new_list.append(tar)

                

            
print(new_list)
# (1 - score) because lesser metric is better

score1 = 1. - log_loss_metric(train_targets, res_NN)

score2 = 1. - 0.014921110834529303

scores = [score1,score2]

sum_scores = sum(scores)



weights = [x / sum_scores for x in scores]
ss_NN.loc[:, train_targets.columns[avg_list]] =  ss_NN.loc[:, train_targets.columns[avg_list]] * weights[0] +  sub.loc[:, train_targets.columns[avg_list]] * weights[1]

res_NN.loc[:, train_targets.columns[avg_list]] = res_NN.loc[:, train_targets.columns[avg_list]] * weights[0]  +  valid_results.loc[:, train_targets.columns[avg_list]] * weights[1]
print(f'pytorch OOF Metric with postprocessing: {log_loss_metric(train_targets, valid_results)}')
print(f'Final OOF Metric with postprocessing: {log_loss_metric(train_targets, res_NN)}')
submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



public_ids = list(submission['sig_id'].values)



submission.shape
test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



test_ids = list(test['sig_id'].values)



private_ids = list(set(test_ids)-set(public_ids))



len(private_ids)
predictions = [ss_NN,sub]

len(predictions)
target_columns = list(submission.columns)

target_columns.remove('sig_id')
y_pred = pd.DataFrame()

y_pred['sig_id'] = predictions[0]['sig_id']



for column in target_columns:

    column_data = []

    for i in range(len(predictions)):

        column_data.append(predictions[i][column])

    y_pred[column] = np.mean(column_data, axis=0)



y_pred.shape
# (1 - score) because lesser metric is better

score1 = 1. - 0.01866

score2 = 1. - 0.01867

scores = [score1,score2]

sum_scores = sum(scores)



weights = [x / sum_scores for x in scores]
weighted_y_pred = pd.DataFrame()

weighted_y_pred['sig_id'] = predictions[0]['sig_id']



for column in target_columns:

    column_data = []

    for i in range(len(predictions)):

        column_data.append(predictions[i][column] * weights[i])

    weighted_y_pred[column] = np.sum(column_data, axis=0)



weighted_y_pred.shape
y_pred = weighted_y_pred
submission = pd.DataFrame(index = public_ids + private_ids, columns=target_columns)

submission.index.name = 'sig_id'



submission[:] = 0



submission.loc[y_pred.sig_id,:] = y_pred[target_columns].values



submission.loc[test[test.cp_type == 'ctl_vehicle'].sig_id] = 0



submission.to_csv('submission.csv', index=True)



submission.head().T