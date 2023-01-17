#!pip install iterative-stratification
import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master/')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import tensorflow_addons as tfa

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss

from tqdm.notebook import tqdm

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
#def preprocess(df):

#    df = df.copy()

#    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp':0, 'ctl_vehicle':1})

#    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1':0, 'D2':1})

#    del df['sig_id']

#    return df
####Drop ID's

train_targets=train_targets.drop(train_targets.columns[0],axis=1)

train_targets.head()
####Drop ID's but Save Test ID's

test_id=test_features['sig_id']

test_features=test_features.drop(test_features.columns[0],axis=1)

test_features.head()
####Drop ID's

train_features =train_features.drop(train_features.columns[0],axis=1)

train_features.head() 
train_features.iloc[:,3:]
test_features
train_targets['total']=train_targets.sum(axis=1)

train_targ_control=train_targets[train_targets['total']==0]

train_targ_control_index=train_targ_control.index.values.tolist() 

del train_targets['total']
train_features_control=train_features.iloc[train_targ_control_index]
control_stats=pd.DataFrame(train_features_control.describe())

control_col_25=pd.DataFrame(control_stats.iloc[4])

control_col_75=pd.DataFrame(control_stats.iloc[6])

control_stats
adj_features=train_features.iloc[:,3:]

adj_features
adj_test_features=test_features.iloc[:,3:]

adj_test_features
adj_test_features_trans=adj_test_features.T
adj_features_trans=adj_features.T
control_col_25=control_col_25.iloc[1:,:]
control_col_75=control_col_75.iloc[1:,:]
control_col_75.iloc[1][0]
(adj_test_features_trans.iloc[0]>control_col_25.iloc[0][0]) & (adj_test_features_trans.iloc[0]<control_col_75.iloc[0][0])
#(adj_test_features_trans.iloc[rowindex]>control_col_25.iloc[colindex][0]) & (adj_test_features_trans.iloc[rowindex]<control_col_75.iloc[colindex][0])
#adj_test_features_trans.iloc[0]
#adj_test_features.iloc[:,0][(adj_test_features_trans.iloc[0]>control_col_25.iloc[0][0]) & (adj_test_features_trans.iloc[0]<control_col_75.iloc[0][0])]=0
####Iterate through rows removing activity within 25 to 75 percentile for each gene

for i in range(872):

    adj_test_features.iloc[:,i][(adj_test_features_trans.iloc[i]>control_col_25.iloc[i][0]) & (adj_test_features_trans.iloc[i]<control_col_75.iloc[i][0])]=0
####Iterate through training set

for i in range(872):

    adj_features.iloc[:,i][(adj_features_trans.iloc[i]>control_col_25.iloc[i][0]) & (adj_test_features_trans.iloc[i]<control_col_75.iloc[i][0])]=0
adj_test_features
adj_features
train_features=train_features.iloc[:,:3].join(adj_features)

train_features
test_features=test_features.iloc[:,:3].join(adj_test_features)

test_features
train_features=train_features[train_features['cp_type'] != 'ctl_vehicle']
indexs_list2=train_features.index.values.tolist() 
train_targets=train_targets.iloc[indexs_list2]
###Reindex

train_features = train_features.reset_index()

del train_features['index']

train_features
###Reindex

train_targets = train_targets.reset_index()

del train_targets['index']

train_targets
filter_col_g = [col for col in train_features if col.startswith('g-')]

genes=train_features[filter_col_g]

genes.head()
filter_col_c = [col for col in train_features if col.startswith('c-')]

cells=train_features[filter_col_c]

cells.head()
filter_col_c_test = [col for col in test_features if col.startswith('c-')]

cells_test=test_features[filter_col_c_test]

cells_test.head()
filter_col_g_test = [col for col in test_features if col.startswith('g-')]

genes_test=test_features[filter_col_g_test]

genes_test.head()
from sklearn.decomposition import PCA

###Add PCA Features###

pca_c = PCA(.9)

pca_g = PCA(.9)



#fit PCA on Training Set

pca_c.fit(cells)

pca_g.fit(genes)



### Apply PCA Mapping to Training and Test Set: Converts to a np.array

pca_cells_train = pca_c.transform(cells)

pca_genes_train = pca_g.transform(genes)

pca_cells_test = pca_c.transform(cells_test)

pca_genes_test = pca_g.transform(genes_test)



#####Create Dataframe of PCA Features

PCA_g_train=pd.DataFrame(pca_genes_train)

PCA_c_train=pd.DataFrame(pca_cells_train)

PCA_g_test=pd.DataFrame(pca_genes_test)

PCA_c_test=pd.DataFrame(pca_cells_test)
PCA_g_train = PCA_g_train.reset_index()

del PCA_g_train['index']



PCA_c_train = PCA_c_train.reset_index()

del PCA_c_train['index']



PCA_g_test = PCA_g_test.reset_index()

del PCA_g_test['index']



PCA_c_test = PCA_c_test.reset_index()

del PCA_c_test['index']
print(PCA_g_train.shape)

print(PCA_c_train.shape)

print(PCA_g_test.shape)

print(PCA_c_test.shape)

print(test_features.shape)

print(train_features.shape)
PCA_train=pd.merge(PCA_g_train, PCA_c_train,right_index=True, left_index=True)

PCA_test=pd.merge(PCA_g_test, PCA_c_test,right_index=True, left_index=True)
####One Hot Code Train Columns: cp_type and cp_dose

dummies=train_features[['cp_type','cp_dose']]

cat_columns = ['cp_type','cp_dose']
dummies2=pd.get_dummies(dummies, prefix_sep="_",

                              columns=cat_columns)

dummies2
del train_features['cp_type']

del train_features['cp_dose']

from sklearn.preprocessing import PowerTransformer

# perform a yeo-johnson transform of the dataset

pt = PowerTransformer(method='yeo-johnson')

data1 = pt.fit_transform(train_features)

# convert the array back to a dataframe

train_features = pd.DataFrame(data1)
train_features.insert(loc=0, column='cp_type', value=dummies2['cp_type_trt_cp'])

train_features.insert(loc=2, column='cp_dose', value=dummies2['cp_dose_D1'])

#train_features['cp_type']=dummies2['cp_type_trt_cp']

#train_features['cp_dose']=dummies2['cp_dose_D1']
####One Hot Code Columns: cp_type and cp_dose

dummies3=test_features[['cp_type','cp_dose']]
dummies4=pd.get_dummies(dummies3, prefix_sep="_",

                              columns=cat_columns)

dummies4
del test_features['cp_type']

del test_features['cp_dose']
# perform a yeo-johnson transform of the dataset

pt = PowerTransformer(method='yeo-johnson')

data2 = pt.fit_transform(test_features)

# convert the array back to a dataframe

test_features = pd.DataFrame(data2)
test_features.insert(loc=0, column='cp_type', value=dummies4['cp_type_trt_cp'])

test_features.insert(loc=2, column='cp_dose', value=dummies4['cp_dose_D1'])

#test_features['cp_type']=dummies4['cp_type_trt_cp']

#test_features['cp_dose']=dummies4['cp_dose_D1']
test_cont=test_features['cp_type'] == 0

test_cont
train_targets = train_targets.reset_index()

del train_targets['index']

train_targets
#import numpy as np

#sys.path.append('../input/hellinger2/hellinger_distance_criterion.pyx')

#import hellinger_distance_criterion.pyx

#from sklearn.ensemble import RandomForestClassifier



#hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64'))

#clf = RandomForestClassifier(criterion=hdc, max_depth=4, n_estimators=100)

#clf.fit(X_train, y_train)

#print('hellinger distance score: ', clf.score(X_test, y_test))
def create_model(num_columns):

    model = tf.keras.Sequential([

    tf.keras.layers.Input(num_columns), 

    tf.keras.layers.BatchNormalization(), 

    tf.keras.layers.Dropout(0.2), 

    tf.keras.layers.Dense(800, activation="swish"),

    #2048

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(400, activation="swish"),

    #1048

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(206, activation="sigmoid")

    ])

  

    model.compile(optimizer=tf.optimizers.Adam(),

                  loss='binary_crossentropy')

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
top=train_features.iloc[:, top_feats]

top
top.shape
def metric(y_true, y_pred):

    metrics = []

    for _target in train_targets.columns:

        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels=[0,1]))

    return np.mean(metrics)
train_targets
print(train_features.shape)

print(test_features.shape)

print(PCA_train.shape)

print(PCA_test.shape)
test_features=test_features.iloc[:, top_feats]

train_features=train_features.iloc[:, top_feats]
test_features = test_features.reset_index()

del test_features['index']



train_features = train_features.reset_index()

del train_features['index']
train_features=pd.merge(train_features, PCA_train,right_index=True, left_index=True)
test_features=pd.merge(test_features, PCA_test,right_index=True, left_index=True)

train=train_features.copy()

train
test=test_features.copy()
input_dim = train.shape[1]
N_STARTS = 3

# tensorflow

tf.random.set_seed(42)



res = train_targets.copy()



ss.loc[:, train_targets.columns] = 0

res.loc[:, train_targets.columns] = 0





for seed in range(N_STARTS):

    for n, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=5, random_state=seed, shuffle=True).split(train_targets, train_targets)):

        print(f'Fold {n}')

        

        model = create_model(input_dim)

        checkpoint_path = f'repeat:{seed}_Fold:{n}.hdf5'



        reduce_lr_loss = ReduceLROnPlateau(

            monitor='val_loss',

            factor=0.1, patience=3,

            verbose=1,

            epsilon=1e-4,

            mode='min'

        )

        



        cb_checkpt = ModelCheckpoint(

            checkpoint_path,

            monitor='val_loss',

            verbose=0,

            save_best_only=True,

            save_weights_only=True,

            mode='min'

        )

        

        model.fit(

            train.values[tr],

            train_targets.values[tr],

            validation_data = (train.values[te], train_targets.values[te]),

            epochs=40,

            batch_size=128,

            callbacks=[reduce_lr_loss, cb_checkpt],

            verbose=2

                 )

        

        model.load_weights(checkpoint_path)

        test_predict = model.predict(test.values)

        val_predict = model.predict(train.values[te])

        

        ss.loc[:, train_targets.columns] += test_predict

        res.loc[te, train_targets.columns] += val_predict

        print('')

        

ss.loc[:, train_targets.columns] /= ((n+1) * N_STARTS)

res.loc[:, train_targets.columns] /= N_STARTS
print(f'OOF Metric: {metric(train_targets, res)}')
ss.loc[test_cont, train_targets.columns] = 0
ss.to_csv('submission.csv', index=False)