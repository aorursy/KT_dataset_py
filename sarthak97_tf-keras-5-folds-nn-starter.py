from IPython.display import HTML

HTML('<center><iframe width="560" height="315" src="https://www.youtube.com/embed/PGzT3cTPah8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import random



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint

import tensorflow_addons as tfa



from sklearn.metrics import log_loss

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
# Seed Everythig !!

def seed_everything(seed=2020): 

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)
SEED = 42

NFOLDS = 5

EPOCHS = 50

BATCH_SIZE = 128

ID = 'sig_id'

root = '../input/lish-moa/'

seed_everything(SEED)
train = pd.read_csv(root + 'train_features.csv')

target = pd.read_csv(root + 'train_targets_scored.csv')

test = pd.read_csv(root + 'test_features.csv')
train.sample(5)
target.sample(5)
test.sample(5)
sub_df = pd.read_csv(root + 'sample_submission.csv')

sub_df.sample(5)
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
# keep the ID column separate

train_id = train[ID]

test_id = test[ID]

sub_id = sub_df[ID]
#Keeping the important features only

important_cols = []

train_cols = train.columns

for i in range(len(train_cols)):

    if i in top_feats:

        important_cols.append(train_cols[i])

print(len(important_cols))



train = train[important_cols]

test = test[important_cols]
def preprocess(df):

    _df = df.copy()

    _df['cp_type'] = _df['cp_type'].apply(lambda x : 1 if x == 'ctl_vehicle' else 0)

    _df['cp_dose'] = _df['cp_dose'].apply(lambda x : 1 if x == 'D2' else 0)

    return _df



train = preprocess(train)

test = preprocess(test)



del target[ID]



target = target.loc[train['cp_type']==0].reset_index(drop=True)

train = train.loc[train['cp_type']==0].reset_index(drop=True)
# ===== SCALING ===== #



train['WHERE'] = 'train'

test['WHERE'] = 'test'

temp = train.append(test)



scaler = StandardScaler()

temp.iloc[:, :-1] = scaler.fit_transform(temp.iloc[:, :-1])
temp.sample(5)
# separate train and test data

train = temp.loc[temp.WHERE == 'train']

test = temp.loc[temp.WHERE == 'test']

del train['WHERE']

del test['WHERE']
# Define callbacks



def get_early_stopper():

    earlyStop = EarlyStopping( monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto',

        baseline=None, restore_best_weights=True)

    return earlyStop





def get_lr_callback(batch_size = 64, plot = False):

    """Returns a lr_scheduler callback which is used for training.

    Feel free to change the values below!

    """

    lr_start   = 0.001

    lr_max     = 0.001 * BATCH_SIZE # higher batch size --> higher lr

    lr_min     = 0.00001

    # 30% of all epochs are used for ramping up the LR and then declining starts

    lr_ramp_ep = EPOCHS * 0.3

    lr_sus_ep  = 0

    lr_decay   = 0.9



    def lr_scheduler(epoch):

            if epoch < lr_ramp_ep:

                lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start



            elif epoch < lr_ramp_ep + lr_sus_ep:

                lr = lr_max



            else:

                lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min



            return lr

    

    if plot == False:

        # get the Keras-required callback with our LR for training

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,verbose = 2)

        return lr_callback 

    

    else: 

        return lr_scheduler



    

def reduce_lr_on_plateau():

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,

                                  epsilon = 1e-4, mode = 'min', verbose=1)

    return reduce_lr
def make_model(l): 

    model = tf.keras.Sequential([

        Input(l),

        BatchNormalization(),

        Dropout(0.2),

        tfa.layers.WeightNormalization(Dense(2048, activation="elu")),

        BatchNormalization(),

        Dropout(0.5),

        tfa.layers.WeightNormalization(Dense(1024, activation="elu")),

        BatchNormalization(),

        Dropout(0.5),

        tfa.layers.WeightNormalization(Dense(206, activation="sigmoid")),

    ])



    model.compile(loss='binary_crossentropy',

                 #optimizer = tf.keras.optimizers.Adam(),

                 #optimizer = tfa.optimizers.LazyAdam(0.001),

                 optimizer = tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),

                 #optimizer = tf.keras.optimizers.SGD(),

                  metrics=["accuracy"]

                 )

    return model
net = make_model(len(train.columns))

net.summary()
y_features = target.columns

_input = train.shape[1]

oof_preds = np.zeros((train.shape[0], 206))

sub_df.loc[:, y_features] = 0



train = train.values

target = target.values

test = test.values
N_START = 7

tf.random.set_seed(SEED)



for seed in range(N_START):

    print(f"\nSEED {seed+1}\n")

    for fold, (tr_idx, val_idx) in enumerate(MultilabelStratifiedKFold(n_splits=NFOLDS, random_state=42, 

                                                                       shuffle=True).split(train, target)):

        print(f"\nFOLD {fold+1}\n")

        X_train = train[tr_idx]

        X_val = train[val_idx]

        y_train = target[tr_idx]

        y_val = target[val_idx]



        net = make_model(_input)



        checkpoint_path = f'repeat:{seed}_Fold:{fold}.hdf5'

        cb_checkpt = ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True,

                                         save_weights_only = True, mode = 'min')



        net.fit(X_train, y_train,

                batch_size = BATCH_SIZE,

                epochs = EPOCHS,

                validation_data = (X_val, y_val),

                #callbacks = [get_early_stopper(), get_lr_callback(BATCH_SIZE)],

                callbacks = [get_early_stopper(), reduce_lr_on_plateau(), cb_checkpt],

                verbose=2

               )



        net.load_weights(checkpoint_path)

        # net.evaluate() returns loss values and metric values

        print("[INFO] Train : ", net.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0, return_dict=True))

        print("[INFO] Validation : ", net.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0, return_dict=True))

        print("[INFO] Predicting val...")

        oof_preds[val_idx] = net.predict(X_val, batch_size=BATCH_SIZE, verbose=0)

        print("[INFO] Predicting test...")

        sub_df.loc[:, y_features] += net.predict(test, batch_size=BATCH_SIZE, verbose=0) / NFOLDS
def score(y_true, y_preds):

    metric = []

    for col in range(target.shape[1]):

        metric.append(log_loss(y_true[:, col], y_preds[:, col].astype('float'), labels=[0,1]))

    return np.mean(metric)
metric = score(target, oof_preds)

print(f"OOF Metric : {metric}")
sub_df.to_csv("submission.csv", index=False)