import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import os
import gc
import datetime
import numpy as np
import pandas as pd
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from tqdm.notebook import tqdm
from time import time
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
ss_lr = ss.copy()

cols = [c for c in ss.columns.values if c != 'sig_id']
def preprocess(df):
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    del df['sig_id']
    return df

# [Fast Numpy Log Loss] https://www.kaggle.com/gogo827jz/optimise-blending-weights-4-5x-faster-log-loss
def log_loss_metric(y_true, y_pred):
    loss = 0
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    for i in range(y_pred.shape[1]):
        loss += - np.mean(y_true[:, i] * np.log(y_pred_clip[:, i]) + (1 - y_true[:, i]) * np.log(1 - y_pred_clip[:, i]))
    return loss / y_pred.shape[1]

train = preprocess(train_features)
test = preprocess(test_features)

del train_targets['sig_id']

train_targets = train_targets.loc[train['cp_type']==0].reset_index(drop=True)
train = train.loc[train['cp_type']==0].reset_index(drop=True)
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
params = {'learning_rate': 0.3, 
          'depth': 6, 
          'l2_leaf_reg': 3, 
          'loss_function': 'MultiRMSE', 
          'eval_metric': 'MultiRMSE', 
          'task_type': 'CPU', 
          'iterations': 200,
          'od_type': 'Iter', 
          'boosting_type': 'Plain', 
          'bootstrap_type': 'Bernoulli', 
          'allow_const_label': True, 
         }
N_STARTS = 1
N_SPILTS = 5

res = train_targets.copy()
ss.loc[:, train_targets.columns] = 0
res.loc[:, train_targets.columns] = 0

for seed in range(N_STARTS):
    
    for n, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits = N_SPILTS, random_state = 42, shuffle = True).split(train_targets, train_targets)):
        
        start_time = time()
        x_tr, x_val = train.values[tr][:, top_feats], train.values[te][:, top_feats]
        y_tr, y_val = train_targets.astype(float).values[tr], train_targets.astype(float).values[te]
        x_tt = test.values[:, top_feats]
        
        cat_tr = Pool(x_tr, label = y_tr)
        cat_val = Pool(x_val, label = y_val)
        
        params['random_state'] = seed
        model = CatBoostRegressor(**params)
        fit_model = model.fit(cat_tr, eval_set = cat_val, early_stopping_rounds = 5, 
                              use_best_model = True, verbose = 0)
        
        ss.loc[:, train_targets.columns] += fit_model.predict(x_tt) / (N_SPILTS * N_STARTS)
        fold_pred = fit_model.predict(x_val)
        res.loc[te, train_targets.columns] += fold_pred / N_STARTS
        fold_score = log_loss_metric(train_targets.loc[te].values, fold_pred)
        print(f'[{str(datetime.timedelta(seconds = time() - start_time))[0:7]}] CatBoost: Seed {seed}, Fold {n}:', fold_score)
        
        del model, fit_model
        x = gc.collect()
print(f'CatBoost OOF Metric: {log_loss_metric(train_targets.values, res.values)}')
X_new = res[cols].values
x_tt_new = ss[cols].values
from sklearn.linear_model import LogisticRegression

N_STARTS = 1
N_SPLITS = 5

res_lr = train_targets.copy()
ss_lr.loc[:, train_targets.columns] = 0
res_lr.loc[:, train_targets.columns] = 0

for tar in tqdm(range(train_targets.shape[1])):
    
    start_time = time()
    targets = train_targets.values[:, tar]
    
    if targets.sum() >= N_SPLITS:
        
        for seed in range(N_STARTS):

            skf = StratifiedKFold(n_splits = N_SPLITS, random_state = 42, shuffle = True)

            for n, (tr, te) in enumerate(skf.split(targets, targets)):

                x_tr, x_val = X_new[tr, tar].reshape(-1, 1), X_new[te, tar].reshape(-1, 1)
                y_tr, y_val = targets[tr], targets[te]
                
                model = LogisticRegression(penalty = 'none', random_state = seed)
                model.fit(x_tr, y_tr)
                ss_lr.loc[:, train_targets.columns[tar]] += model.predict_proba(x_tt_new[:, tar].reshape(-1, 1))[:, 1] / (N_SPLITS * N_STARTS)
                res_lr.loc[te, train_targets.columns[tar]] += model.predict_proba(x_val)[:, 1] / N_STARTS
    
    score = log_loss(train_targets.loc[:, train_targets.columns[tar]].values, res_lr.loc[:, train_targets.columns[tar]].values)
    print(f'[{str(datetime.timedelta(seconds = time() - start_time))[2:7]}] LR Target {tar}:', score)
print(f'LR OOF Metric: {log_loss_metric(train_targets.values, res_lr.values)}')
ss.loc[test['cp_type'] == 1, train_targets.columns] = 0
ss.to_csv('submission_cat.csv', index = False)
ss_lr.loc[test['cp_type'] == 1, train_targets.columns] = 0
ss_lr.to_csv('submission.csv', index = False)