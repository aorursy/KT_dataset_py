!pip install ../input/my-wheels/scikit_learn-0.21.0-cp37-cp37m-manylinux1_x86_64.whl



# !pip install -U scikit-learn==0.21.0

# !pip install scikit-optimize==0.8.dev0
import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os

import gc

import random

import math

import time

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import skopt

from skopt import gp_minimize, forest_minimize

from skopt.space import Real, Categorical, Integer

from skopt.plots import *

from skopt.utils import use_named_args



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import *

from tensorflow.keras.callbacks import *

from tensorflow.keras.models import *

import tensorflow_addons as tfa



import sklearn

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.metrics import log_loss

import category_encoders as ce

from tqdm.notebook import tqdm

print(os.listdir('../input/lish-moa'))
skopt.__version__, sklearn.__version__
FOLDS = 7    # 10

EPOCHS = 65

BATCH_SIZE = 128

# LR = 0.001

WD = 1e-5

VERBOSE = 0



N_TRIALS = 15   # 20

SEED = 34    # 2020



PERM_IMP = False

OHE = False
def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



seed_everything(seed=42)
def preprocess(df):

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})

    del df['sig_id']

    return df

# permuation importance by: https://www.kaggle.com/simakov/keras-multilabel-neural-network-v1-2



# hardoced features

top_feats = [  0,   1,   2,   3,   5,   6,   8,   9,  10,  11,  12,  14,  15,

        16,  18,  19,  20,  21,  23,  24,  25,  27,  28,  29,  30,  31,

        32,  33,  34,  35,  36,  37,  39,  40,  41,  42,  44,  45,  46,

        48,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,

        63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74,  75,  76,

        78,  79,  80,  81,  82,  83,  84,  86,  87,  88,  89,  90,  92,

        93,  94,  95,  96,  97,  99, 100, 101, 103, 104, 105, 106, 107,

       108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,

       121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134,

       135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,

       149, 150, 151, 152, 153, 154, 155, 157, 159, 160, 161, 163, 164,

       165, 166, 167, 168, 169, 170, 172, 173, 175, 176, 177, 178, 180,

       181, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 195,

       197, 198, 199, 202, 203, 205, 206, 208, 209, 210, 211, 212, 213,

       214, 215, 218, 219, 220, 221, 222, 224, 225, 227, 228, 229, 230,

       231, 232, 233, 234, 236, 238, 239, 240, 241, 242, 243, 244, 245,

       246, 248, 249, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260,

       261, 263, 265, 266, 268, 270, 271, 272, 273, 275, 276, 277, 279,

       282, 283, 286, 287, 288, 289, 290, 294, 295, 296, 297, 299, 300,

       301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 315,

       316, 317, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331,

       332, 333, 334, 335, 338, 339, 340, 341, 343, 344, 345, 346, 347,

       349, 350, 351, 352, 353, 355, 356, 357, 358, 359, 360, 361, 362,

       363, 364, 365, 366, 368, 369, 370, 371, 372, 374, 375, 376, 377,

       378, 379, 380, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,

       392, 393, 394, 395, 397, 398, 399, 400, 401, 403, 405, 406, 407,

       408, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422,

       423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435,

       436, 437, 438, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,

       452, 453, 454, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,

       466, 468, 469, 471, 472, 473, 474, 475, 476, 477, 478, 479, 482,

       483, 485, 486, 487, 488, 489, 491, 492, 494, 495, 496, 500, 501,

       502, 503, 505, 506, 507, 509, 510, 511, 512, 513, 514, 516, 517,

       518, 519, 521, 523, 525, 526, 527, 528, 529, 530, 531, 532, 533,

       534, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,

       549, 550, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563,

       564, 565, 566, 567, 569, 570, 571, 572, 573, 574, 575, 577, 580,

       581, 582, 583, 586, 587, 590, 591, 592, 593, 595, 596, 597, 598,

       599, 600, 601, 602, 603, 605, 607, 608, 609, 611, 612, 613, 614,

       615, 616, 617, 619, 622, 623, 625, 627, 630, 631, 632, 633, 634,

       635, 637, 638, 639, 642, 643, 644, 645, 646, 647, 649, 650, 651,

       652, 654, 655, 658, 659, 660, 661, 662, 663, 664, 666, 667, 668,

       669, 670, 672, 674, 675, 676, 677, 678, 680, 681, 682, 684, 685,

       686, 687, 688, 689, 691, 692, 694, 695, 696, 697, 699, 700, 701,

       702, 703, 704, 705, 707, 708, 709, 711, 712, 713, 714, 715, 716,

       717, 723, 725, 727, 728, 729, 730, 731, 732, 734, 736, 737, 738,

       739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751,

       752, 753, 754, 755, 756, 758, 759, 760, 761, 762, 763, 764, 765,

       766, 767, 769, 770, 771, 772, 774, 775, 780, 781, 782, 783, 784,

       785, 787, 788, 790, 793, 795, 797, 799, 800, 801, 805, 808, 809,

       811, 812, 813, 816, 819, 820, 821, 822, 823, 825, 826, 827, 829,

       831, 832, 833, 834, 835, 837, 838, 839, 840, 841, 842, 844, 845,

       846, 847, 848, 850, 851, 852, 854, 855, 856, 858, 860, 861, 862,

       864, 867, 868, 870, 871, 873, 874]



print(len(top_feats))
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')



subm = pd.read_csv('../input/lish-moa/sample_submission.csv')
cat_features = ['cp_type', 'cp_time', 'cp_dose']

num_features = [c for c in train_features.columns if train_features.dtypes[c] != 'object']

num_features = [c for c in num_features if c not in cat_features]

target_cols  = [c for c in train_targets.columns if c not in ['sig_id']]





# filter ctl samples 

train_targets['cp_type'] = train_features['cp_type']

train_features = train_features[train_features['cp_type'] != 'ctl_vehicle']

train_targets = train_targets[train_targets['cp_type'] != 'ctl_vehicle']
train_x = preprocess(train_features)

test_x = preprocess(test_features)



targets = train_targets[target_cols]

print('Tensor shapes:', train_x.shape, targets.shape, test_x.shape)





train_x = train_x.drop(['cp_type'], axis=1)

# train_targets = train_targets.drop(['cp_type'], axis=1)

train_x = train_x.reset_index().drop(['index'], axis=1)

targets = targets.reset_index().drop(['index'], axis=1)





if PERM_IMP: 

    train_x = train_x.iloc[:, top_feats]

    test_x = test_x.iloc[:, top_feats]



# ohe cols

if OHE:

    train_x = pd.get_dummies(train_x, columns=train_x[cat_features].columns)

    test_x = pd.get_dummies(test_x, columns=test_x[cat_features].columns)



print('Final tensor shapes:', train_x.shape, targets.shape, test_x.shape)
train_x.head()
# from sklearn.model_selection import train_test_split



# x_train, x_val, y_train, y_val = train_test_split(train_x, targets, test_size=0.1)

    

# x_train.shape, x_val.shape
# dim_num_dense_layers = Integer(low=2, high=10, name='num_dense_layers')



# dim_learning_rate = Real(low=1e-4, high=1e-1, prior='uniform', name='learning_rate')

# dim_num_dense_nodes_1 = Integer(low=1024, high=2048, name='num_dense_nodes_1')

# dim_num_dense_nodes_2 = Integer(low=512, high=1024, name='num_dense_nodes_2')

dim_num_dense_nodes_3 = Integer(low=256, high=512, name='num_dense_nodes_3')

dim_activation = Categorical(categories=['relu', 'elu'], name='activation')

dim_dropout1 = Integer(low=1, high=5, name='dp1')

dim_dropout2 = Integer(low=1, high=5, name='dp2')

dim_dropout3 = Integer(low=1, high=5, name='dp3')

dim_dropout4 = Integer(low=1, high=5, name='dp4')

dim_smooth = Real(low=0.0001, high=0.005, name='smooth')



# dim_look_ahead = Integer(low=5, high=15, name='look_ahead')





dimensions = [

#     dim_learning_rate,

#     dim_num_dense_nodes_1,

#     dim_num_dense_nodes_2,

    dim_num_dense_nodes_3,

    dim_activation,

    dim_dropout1,

    dim_dropout2,

    dim_dropout3,

    dim_dropout4,

    dim_smooth

]





# set default params - make sure are within the search space

default_params = [512, 'relu', 5, 5, 5, 5, 0.0015]  



assert len(default_params)==len(dimensions), 'Error: check shapes!'
n_inputs = train_x.shape[1] # len(top_feats)

n_outs = len(target_cols)



def create_model(

#     learning_rate, 

#     num_dense_nodes_1, 

#     num_dense_nodes_2,

    num_dense_nodes_3,

    activation, 

    dp1,

    dp2,

    dp3,

    dp4,

    smooth

):

    

    weight_norm = False

    

    inp = tf.keras.layers.Input(n_inputs)

    x = tf.keras.layers.BatchNormalization()(inp)

    x = tf.keras.layers.Dropout(0.2)(x)

    

    x = tf.keras.layers.Dense(2048, activation=activation)(x)

    if weight_norm:

        x = tfa.layers.WeightNormalization(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.1*dp1)(x)

    

    x = tf.keras.layers.Dense(1024, activation=activation)(x)

    if weight_norm:

        x = tfa.layers.WeightNormalization(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.1*dp2)(x)

    

    x = tf.keras.layers.Dense(786, activation=activation)(x)

    if weight_norm:

        x = tfa.layers.WeightNormalization(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.1*dp3)(x)

    

    x = tf.keras.layers.Dense(num_dense_nodes_3, activation=activation)(x)

    if weight_norm:

        x = tfa.layers.WeightNormalization(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.1*dp4)(x)

    

    out = tf.keras.layers.Dense(n_outs, activation="sigmoid")(x)                            

    model = tf.keras.models.Model(inp, out)

    

    # optimizers

    #     opt = tf.keras.optimizers.Adam(lr=1e-3)   # learning_rate

    #     opt = tfa.optimizers.SWA(opt, 100)

    opt = tfa.optimizers.AdamW(weight_decay=1e-5)

    #     opt = tfa.optimizers.Lookahead(opt, sync_period=int(look_ahead))

    

    # compile model

    model.compile(optimizer=opt, 

                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=smooth),  # 'binary_crossentropy', 

                  metrics=[tf.keras.losses.BinaryCrossentropy(), 'AUC'])



    return model
path_best_model = './model.h5'

best_loss = np.inf
@use_named_args(dimensions=dimensions)

def fitness(num_dense_nodes_3, activation, dp1, dp2, dp3, dp4, smooth):   

    

    """

    Hyper-parameters:

    num_dense_nodes:   Number of nodes in layer 4.

    activation:        Activation function for all layers.

    dp:                Dropout rates (x4)

    look_ahead:        Look ahead steps (Adam optimizer)

    """



    # Print the hyper-parameters.

    print('num_dense_nodes layer-4:', num_dense_nodes_3)

    print('activation:',activation)

    print('dropout 1:', dp1*0.1)

    print('dropout 2:', dp2*0.1)

    print('dropout 3:', dp3*0.1)

    print('dropout 4:', dp4*0.1)

    print('smooth param:', smooth)

    print()

    

    splits = MultilabelStratifiedKFold(n_splits = FOLDS).split(train_x, targets)

    

    fold_log_loss = 0

    for fold, (trn_ind, val_ind) in enumerate(splits):

        

        K.clear_session()

        # Create the neural network with these hyper-parameters.

        model = create_model(

            num_dense_nodes_3,

            activation, 

            dp1,

            dp2,

            dp3,

            dp4,

            smooth)

    

        # Create callback-functions

        cbs = [

            tf.keras.callbacks.EarlyStopping('val_binary_crossentropy', patience=10, verbose=1),

            tf.keras.callbacks.ReduceLROnPlateau('val_binary_crossentropy', patience=3, factor=0.1) ]



        # train the model

        hist = model.fit(x = train_x.iloc[trn_ind],

                         y = targets.iloc[trn_ind],

                         epochs=EPOCHS,

                         batch_size=BATCH_SIZE,

                         validation_data=(train_x.iloc[val_ind], targets.iloc[val_ind]),

                         callbacks=cbs,

                         verbose = VERBOSE)

    

    

        # Get the error on the validation-set

        log_loss = min(hist.history['val_binary_crossentropy'])   

        

        fold_log_loss += log_loss



        # Print the classification accuracy.

        print('-'*20)

        print(f"> Fold {fold} Logloss: {log_loss}")

        print('-'*20)



    # Save the model if it improves on the best-found performance.

    global best_loss



    # If the classification accuracy of the saved model is improved ...

    if fold_log_loss < best_loss:

        

        # Save the new model & Update the error

        model.save(path_best_model)

        best_loss = fold_log_loss



    # Delete the Keras model with these hyper-parameters from memory.

    del model

    gc.collect()

    

    # Clear the Keras session, to empty the TensorFlow graph 

    K.clear_session()

    

    return fold_log_loss
# check objective function (uncomment bellow if you like to test)



fitness(default_params)
search_result = skopt.gp_minimize(func=fitness,   

                            dimensions=dimensions,

                            acq_func='EI',    #  'gp_hedge'       

                            n_calls=N_TRIALS,

                            random_state=SEED,

                            x0=default_params)
print('optimal hyper-parameters') 

print()

# print(f'lr: {search_result.x[0]}')

# print(f'dense_units 1: {search_result.x[1]}')

# print(f'dense_units 2: {search_result.x[2]}')

print(f'dense_units 3: {search_result.x[0]}')

print(f'activation: {search_result.x[1]}')

print(f'dropout 1: {search_result.x[2]}')

print(f'dropout 2: {search_result.x[3]}')

print(f'dropout 3: {search_result.x[4]}')

print(f'dropout 4: {search_result.x[5]}')

print(f'Smooth param: {search_result.x[6]}')





# ----------------------------

# optimal hyper-parameters v.3

# ----------------------------

# dense_units 3: 511

# activation: elu

# dropout 1: 3

# dropout 2: 4

# dropout 3: 5

# dropout 4: 2
pd.DataFrame(sorted(zip(search_result.func_vals, search_result.x_iters)), index=np.arange(N_TRIALS), columns=['score', 'params'])
%matplotlib inline

plot_convergence(search_result)
# create a list for plotting

dim_names = ['num_dense_nodes_3', 'activation', 'dropout_1', 'dropout_2', 'dropout_3', 'dropout_4', 'smooth_param']



# %matplotlib inline

plot_objective(result=search_result, dimensions=dim_names);
# create model with best hyperparams



model = create_model(*search_result.x)

model.summary()
FOLDS = 7



oof = targets.copy()

subm.loc[:, target_cols] = 0

oof.loc[:, target_cols] = 0





splits = MultilabelStratifiedKFold(n_splits = FOLDS).split(train_x, targets)

# splits = KFold(n_splits=FOLDS, random_state=SEED, shuffle=True).split(train_x)



for n, (tr, te) in enumerate(splits):

    

    print(f'Fold {n}')    



    checkpoint_path = f'./model_Fold_{n}.h5'

    cbs = [

        tf.keras.callbacks.EarlyStopping('val_loss', patience=10, verbose=0),

        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, restore_best_weights=True, verbose=0,  save_weights_only=True, mode='min'),

        tf.keras.callbacks.ReduceLROnPlateau('val_loss', patience=5, factor=0.1, verbose=0, min_delta=1e-4, mode='min'),

        ]



    model = create_model(*search_result.x)



    model.fit(train_x.values[tr], targets.values[tr],

              validation_data=(train_x.values[te], targets.values[te]),

              epochs=EPOCHS, 

              batch_size=BATCH_SIZE, 

              callbacks=cbs, 

              verbose=VERBOSE)



    model.load_weights(checkpoint_path)



    subm.loc[:, target_cols] += model.predict(test_x.values)

    oof.loc[te, target_cols] += model.predict(train_x.values[te])

    print('-'*20)



subm.loc[:, target_cols] /= FOLDS
metrics = []

for _target in target_cols:

    metrics.append(log_loss(targets.loc[:, _target], oof.loc[:, _target]))

    

print(f'OOF Metric: {np.mean(metrics).round(8)}')
# with post-process

oof.loc[train_x['cp_type']==1, target_cols] = 0



metrics = []

for _target in target_cols:

    metrics.append(log_loss(targets.loc[:, _target], oof.loc[:, _target]))



print(f'OOF Metric (with post-processing): {np.mean(metrics).round(8)}')
# # train the model with best hyperparams - all data



# hist = model.fit(train_x, targets,

#                  epochs=100,

#                  batch_size=128,

#                  validation_split=0.2,  # validation_data=(x_val, y_val),

#                  callbacks=cbs)



# predictions = model.predict(test_x.values)

# subm.loc[:,target_cols] = predictions
subm.loc[test_x['cp_type']==1, target_cols] = 0

subm.to_csv('submission.csv', index=False)

print('> Submission saved!')
subm.head()