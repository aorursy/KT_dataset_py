import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
color = sns.color_palette()

from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
#import xgboost as xgb
import lightgbm as lgb
from sklearn import preprocessing as pp 
from scipy.stats import pearsonr 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss 
from sklearn.metrics import precision_recall_curve, average_precision_score 
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report 
DATA_DIR  = os.path.join('/kaggle/input/lish-moa')
TRAIN_FEATURE_FILE = os.path.join(DATA_DIR, 'train_features.csv')
TEST_FEATURE_FILE = os.path.join(DATA_DIR, 'test_features.csv')

TRAIN_TRAGET_FILE = os.path.join(DATA_DIR, 'train_targets_scored.csv')
SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')
train_feat = pd.read_csv(TRAIN_FEATURE_FILE)
train_target = pd.read_csv(TRAIN_TRAGET_FILE)
y_columns = train_target.drop(columns='sig_id', axis=0)
test_feat = pd.read_csv(TEST_FEATURE_FILE)
train_feat.at[train_feat['cp_type'].str.contains('ctl_vehicle'),train_feat.filter(regex='-.*').columns] = 0.0
test_feat.at[test_feat['cp_type'].str.contains('ctl_vehicle'),test_feat.filter(regex='-.*').columns] = 0.0  

for feature in ['cp_type', 'cp_dose']:
    le = LabelEncoder()
    le.fit(list(train_feat[feature].astype(str).values) + list(test_feat[feature].astype(str).values))
    train_feat[feature] = le.transform(list(train_feat[feature].astype(str).values))
    test_feat[feature] = le.transform(list(test_feat[feature].astype(str).values))   

params_lightGB = {
    'num_leaves': 491,
    'min_child_weight': 0.03,
    'feature_fraction': 0.3,
    'bagging_fraction': 0.4,
    'min_data_in_leaf': 106,
    'objective': 'binary',
    'max_depth': -1,
    'learning_rate': 0.01,
    "boosting_type": "gbdt",
    "bagging_seed": 11,
    "metric": 'binary_logloss',
    "verbosity": 0,
    'reg_alpha': 0.4,
    'reg_lambda': 0.6,
    'random_state': 2020
}
skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
sub = pd.read_csv(SUBMISSION_FILE)
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
X_train = train_feat.drop(columns=['sig_id'])
X_test = test_feat.drop(columns=['sig_id'])

# get top_feat    
X_train = X_train.iloc[:, top_feats]
X_test = X_test.iloc[:, top_feats]    

print(X_train.head())
print(X_test.head())
from sklearn.decomposition import PCA

n_components = 600
whiten = False
random_state = 2020

pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)

X_train_PCA = X_train_PCA[[0, 1, 2]]

X_test_PCA = pca.transform(X_test)
X_test_PCA = pd.DataFrame(data=X_test_PCA, index=X_test.index)

X_test_PCA = X_test_PCA[[0, 1, 2]]
X_train = pd.concat([X_train, X_train_PCA], axis=1).copy()
X_test = pd.concat([X_test, X_test_PCA], axis=1).copy()
for i, column in enumerate(y_columns.columns):

    y_train = train_target[column]
    # y_test = ###
    
    trainingScores = []
    cvScores = []
        
    print('                               ')
    print('#######################################')
    print('### start run ', column, ' index ', i )
           
    preds = np.zeros(test_feat.shape[0])
    oof = np.zeros(X_train.shape[0])
           
    for train_idx, val_idx in skf.split(X_train, y_train):


        X_train_fold, X_val_fold = X_train.iloc[train_idx,:], X_train.iloc[val_idx,:]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]


        lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
        lgb_eval = lgb.Dataset(X_val_fold, y_val_fold, reference=lgb_train)
        gbm = lgb.train(params_lightGB, lgb_train, 10000, valid_sets = [lgb_train, lgb_eval], verbose_eval=0, early_stopping_rounds=30)
        
        #loglossTraining = log_loss(y_train_fold, gbm.predict(X_train_fold, num_iteration=gbm.best_iteration), labels=[1,0])
        #trainingScores.append(loglossTraining)

    
        loglossCV = log_loss(y_val_fold, gbm.predict(X_val_fold, num_iteration=gbm.best_iteration), labels=[1,0])       
        cvScores.append(loglossCV)

        preds += gbm.predict(X_test) / skf.n_splits

        oof[val_idx] = gbm.predict(X_train.iloc[val_idx])
        
        #print('Training Log Loss: ', loglossTraining)
        #print('CV Log Loss: ', loglossCV)
            
    loss = log_loss(y_train, oof)
    sub[column] = preds
    
    print('CV Log Loss: ' , loss)
    
    #print('Training Log Loss: ', trainingScores)
    #print('CV Log Loss: ', cvScores)
sub.to_csv('submission.csv', index=False) 
print('done ...')
